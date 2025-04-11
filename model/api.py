"""
Phishing URL Detection API

This Flask application serves a machine learning model that predicts whether a given URL is phishing or safe.
It performs various preprocessing and feature extraction tasks on the URL before making predictions using a pre-trained Random Forest model.

Key functionalities:
- URL cleaning: The script removes shortened URLs using a list of known URL shorteners. It normalizes URLs by:
  - Converting the URL to lowercase.
  - Removing the "www." prefix from the domain name.
  - Stripping query parameters and fragments.
- Feature extraction: The following features are extracted from the URL:
  - URL length, subdomains, and domain length.
  - SSL certificate details (validity, key size, common name mismatch).
  - Suspicious keywords in the URL.
  - DNS records (A, MX, NS records).
- Model prediction: The script uses a pre-trained Random Forest model to predict if the URL is phishing or safe.
  - The model has been trained on various URL features and is loaded using `joblib`.
- One-Hot Encoding: The `ssl_issuer` feature is one-hot encoded using a pre-trained OneHotEncoder.

Endpoints:
- POST /predict: Accepts a JSON payload with a "url" key and returns a prediction ("phishing" or "safe").

Note:
- This script uses Flask for API deployment and provides a development server for testing.
- Production-ready WSGI server (e.g., Gunicorn or uWSGI) must be used when deploying to production.
"""

from flask import Flask, request, jsonify
import joblib
import re
import pandas as pd
import requests
import socket
import ssl
import dns.resolver
import tldextract
import zipfile
import os
from urllib.parse import urlparse, urlunparse
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa, ec
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import gc

app = Flask(__name__)

# Lazy loading for models
model = None
ohe = None
expected_encoded_cols = None

def load_models():
    """Lazy load models only when needed"""
    global model, ohe, expected_encoded_cols
    
    if model is None:
        model_zip = 'random_forest_model.zip'
        extracted_model = 'random_forest_model.pkl'
        
        # Extract the model if not already extracted
        if not os.path.exists(extracted_model):
            with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
        
        # Load the pre-trained model and one-hot encoder
        model = joblib.load(extracted_model)
        ohe = joblib.load('onehot_encoder.pkl')
        
        # Extract expected one-hot encoded column names
        expected_encoded_cols = ohe.get_feature_names_out(['ssl_issuer'])

# URL Cleaning Functions
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
    "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", 
    "l.ead.me", "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", 
    "sniply.io", "short.io", "qrco.de"
})

# Create a smaller thread pool with fewer workers to reduce memory overhead
executor = ThreadPoolExecutor(max_workers=2)

def is_shortened_url(url):
    parsed = urlparse(url)
    domain_cleaned = parsed.netloc.lower().strip()
    return domain_cleaned in SHORTENERS or bool(re.fullmatch(r"/[A-Za-z0-9_-]{4,10}", parsed.path))

# Limit cache size to reduce memory usage
@lru_cache(maxsize=50)
def expand_short_url(url):
    if not is_shortened_url(url):
        return url
    try:
        # Use a session with fewer default settings
        with requests.Session() as session:
            # Disable keep-alive to release connections immediately
            session.headers['Connection'] = 'close'
            # Add timeout to prevent hanging connections
            response = session.head(url, allow_redirects=True, timeout=1)
            return response.url
    except requests.RequestException:
        return None

def clean_url(url):
    url = expand_short_url(url)
    if url is None:
        return None
    parsed = urlparse(url.lower())
    netloc = parsed.netloc.replace("www.", "")
    cleaned = urlunparse((parsed.scheme, netloc, parsed.path.split("?")[0], "", "", ""))
    return cleaned.rstrip("/")

# Reduce DNS cache size
@lru_cache(maxsize=50)
def get_dns_records(domain, record_type):
    try:
        answers = dns.resolver.resolve(domain, record_type)
        return len(answers)
    except Exception:
        return 0

def get_ssl_info(domain):
    ssl_info = {"ssl_valid_days": -1, "ssl_common_name_mismatch": -1, "ssl_self_signed": -1, "ssl_key_size": -1}
    try:
        # Create a new context for each connection to avoid resource leakage
        ctx = ssl.create_default_context()
        s = socket.socket()
        s.settimeout(3)  # Reduced timeout
        ctx_socket = ctx.wrap_socket(s, server_hostname=domain)
        
        try:
            ctx_socket.connect((domain, 443))
            cert = x509.load_der_x509_certificate(ctx_socket.getpeercert(binary_form=True))
            ssl_info.update({
                "ssl_valid_days": (cert.not_valid_after_utc - cert.not_valid_before_utc).days,
                "ssl_key_size": cert.public_key().key_size if isinstance(cert.public_key(), (rsa.RSAPublicKey, ec.EllipticCurvePublicKey)) else -1,
                "ssl_self_signed": int(cert.issuer == cert.subject),
                "ssl_common_name_mismatch": int(cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value != domain)
            })
        finally:
            # Ensure we always close the socket
            ctx_socket.close()
    except Exception:
        pass
    return ssl_info

# Pre-compile regex pattern for IP detection
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# Common suspicious keywords 
SUSPICIOUS_KEYWORDS = frozenset(["login", "verify", "account", "update", "secure", "bank", "confirm", "activate", "signin", "billing"])

def extract_features(url):
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    domain = f"{domain_info.domain}.{domain_info.suffix}"

    # Add default ssl_issuer in case it is not found
    ssl_issuer = 'Unknown'
    
    ssl_info = get_ssl_info(domain)

    # Use more efficient string operations
    url_lower = url.lower()
    path = parsed_url.path
    
    # Count suspicious keywords more efficiently
    suspicious_count = sum(url_lower.count(k) for k in SUSPICIOUS_KEYWORDS)
    
    # Prepare feature dictionary - only calculate what's needed
    features = {
        "ssl_issuer": ssl_issuer,
        "url_length": len(url),
        "num_subdomains": parsed_url.netloc.count('.'),
        "domain_length": len(domain_info.domain),
        "https": int(parsed_url.scheme == "https"),
        "ip_in_url": int(bool(IP_PATTERN.match(parsed_url.netloc))),
        "path_length": len(path),
        "num_special_chars": sum(url.count(ch) for ch in "@?-=_&% ."),
        "num_digits": sum(c.isdigit() for c in url),
        "tld_in_path": int(domain_info.suffix in path),
        "dot_count_in_path": path.count('.'),
        "num_slashes_in_path": path.count('/'),
        "suspicious_keyword": suspicious_count,
        "num_dns_records": get_dns_records(domain, "A"),
        "mx_record_exists": int(get_dns_records(domain, "MX") > 0),
        "num_ns_records": get_dns_records(domain, "NS"),
    }
    
    # Merge SSL info into features
    features.update(ssl_info)

    return features

FEATURE_ORDER = [
    "url_length", "num_subdomains", "domain_length", "https", "ip_in_url", "path_length",
    "num_special_chars", "num_digits", "tld_in_path", "dot_count_in_path", "num_slashes_in_path",
    "suspicious_keyword", "num_dns_records", "mx_record_exists", "num_ns_records",
    "ssl_valid_days", "ssl_common_name_mismatch", "ssl_self_signed", "ssl_key_size"
]

@app.route('/predict', methods=['POST'])
def predict():
    # Load models only when needed
    if model is None:
        load_models()
    
    data = request.get_json(force=True)
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Clean the URL
    cleaned = clean_url(url)
    if cleaned is None:
        return jsonify({"error": "URL cleaning failed"}), 400

    # Extract features from the cleaned URL
    features_dict = extract_features(cleaned)
    
    # Use a more memory-efficient approach instead of pandas
    # Create a feature array directly
    X = np.zeros((1, len(FEATURE_ORDER) + len(expected_encoded_cols)))
    
    # Fill in numerical features
    for i, feature in enumerate(FEATURE_ORDER):
        X[0, i] = features_dict.get(feature, 0)
    
    # Handle categorical features (ssl_issuer)
    ssl_issuer = features_dict.get('ssl_issuer', 'Unknown')
    one_hot = ohe.transform([[ssl_issuer]])
    
    # Add one-hot encoded features to the array
    for i, val in enumerate(one_hot[0]):
        X[0, len(FEATURE_ORDER) + i] = val
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Force garbage collection to release memory
    gc.collect()
    
    return jsonify({"result": "phishing" if prediction == 1 else "safe"})

# Add a route to explicitly unload models when needed
@app.route('/unload_models', methods=['POST'])
def unload_models():
    global model, ohe, expected_encoded_cols
    model = None
    ohe = None
    expected_encoded_cols = None
    gc.collect()
    return jsonify({"status": "Models unloaded successfully"})

# if __name__ == '__main__':
#     app.run(debug=True)
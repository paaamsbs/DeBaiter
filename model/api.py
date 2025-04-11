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
from urllib.parse import urlparse, urlunparse
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa, ec
import numpy as np
from functools import lru_cache
import gc

app = Flask(__name__)

# Load the pre-trained model and one-hot encoder when needed (lazy loading)
model = None
ohe = None
expected_encoded_cols = None

def load_model_if_needed():
    global model, ohe, expected_encoded_cols
    if model is None:
        model = joblib.load('random_forest_model.pkl')
    if ohe is None:
        ohe = joblib.load('onehot_encoder.pkl')
        # Extract expected one-hot encoded column names
        expected_encoded_cols = ohe.get_feature_names_out(['ssl_issuer'])

# URL Cleaning Functions - Use set for O(1) lookups
SHORTENERS = frozenset({"bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
              "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", "l.ead.me",
              "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", "sniply.io", "short.io", "qrco.de"})

def is_shortened_url(url):
    parsed = urlparse(url)
    domain_cleaned = parsed.netloc.lower().strip()
    return domain_cleaned in SHORTENERS or bool(re.fullmatch(r"/[A-Za-z0-9_-]{4,10}", parsed.path))

# Cache results for the same URL but with smaller cache size
@lru_cache(maxsize=50)
def expand_short_url(url):
    if not is_shortened_url(url):
        return url
    try:
        # Use a smaller timeout to prevent hanging connections
        response = requests.head(url, allow_redirects=True, timeout=0.8)
        return response.url
    except requests.RequestException:
        return None

def clean_url(url):
    url = expand_short_url(url)
    if url is None:
        return None
    parsed = urlparse(url.lower())
    netloc = parsed.netloc.replace("www.", "")
    # Avoid unnecessary string operations
    path = parsed.path.split("?")[0] if "?" in parsed.path else parsed.path
    cleaned = urlunparse((parsed.scheme, netloc, path, "", "", ""))
    return cleaned.rstrip("/")

# Feature Extraction Functions with reduced cache size
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
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)  # Reduce timeout to prevent memory leaks from hanging connections
            s.connect((domain, 443))
            cert = x509.load_der_x509_certificate(s.getpeercert(binary_form=True))
            
            # Compute values only if needed
            ssl_info["ssl_valid_days"] = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
            
            # Check key type before accessing key_size to prevent errors
            if isinstance(cert.public_key(), (rsa.RSAPublicKey, ec.EllipticCurvePublicKey)):
                ssl_info["ssl_key_size"] = cert.public_key().key_size
                
            ssl_info["ssl_self_signed"] = int(cert.issuer == cert.subject)
            
            # Get common name only if it exists
            try:
                common_name = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
                ssl_info["ssl_common_name_mismatch"] = int(common_name != domain)
            except (IndexError, AttributeError):
                ssl_info["ssl_common_name_mismatch"] = 1
    except Exception:
        pass
    return ssl_info

# Precompile regex pattern for better performance
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
# Predefine suspicious keywords as a set for O(1) lookups
SUSPICIOUS_KEYWORDS = frozenset(["login", "verify", "account", "update", "secure", "bank", "confirm", "activate", "signin", "billing"])

def extract_features(url):
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    domain = f"{domain_info.domain}.{domain_info.suffix}"

    # Add default ssl_issuer in case it is not found
    ssl_issuer = 'Unknown'
    
    # Get SSL info only once
    ssl_info = get_ssl_info(domain)

    # Use more efficient operations where possible
    path = parsed_url.path
    url_lower = url.lower()
    
    # Count suspicious keywords more efficiently
    suspicious_count = sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in url_lower)

    # Prepare feature dictionary
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
        **ssl_info
    }

    return features

# Define column order for prediction
FEATURE_ORDER = [
    "url_length", "num_subdomains", "domain_length", "https", "ip_in_url", "path_length",
    "num_special_chars", "num_digits", "tld_in_path", "dot_count_in_path", "num_slashes_in_path",
    "suspicious_keyword", "num_dns_records", "mx_record_exists", "num_ns_records",
    "ssl_valid_days", "ssl_common_name_mismatch", "ssl_self_signed", "ssl_key_size"
]

@app.route('/predict', methods=['POST'])
def predict():
    # Load model only when needed
    load_model_if_needed()
    
    try:
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
        
        # Use a single row dataframe instead of a dictionary to avoid conversion overhead
        features_df = pd.DataFrame([features_dict])

        # Handle missing 'ssl_issuer' by adding it with a default value
        if 'ssl_issuer' not in features_df.columns:
            features_df['ssl_issuer'] = 'Unknown'
        
        # One-Hot Encoding for 'ssl_issuer'
        encoded_cat_features = ohe.transform(features_df[['ssl_issuer']])
        
        # Create DataFrame based on the type of output from the encoder
        from scipy import sparse
        if sparse.issparse(encoded_cat_features):
            try:
                # Try pandas >= 1.0.0 syntax
                encoded_cat_df = pd.DataFrame.sparse.from_spmatrix(
                    encoded_cat_features,
                    columns=expected_encoded_cols,
                    index=features_df.index
                )
            except AttributeError:
                # Fallback for older pandas versions
                encoded_cat_df = pd.DataFrame(
                    encoded_cat_features.toarray(),
                    columns=expected_encoded_cols,
                    index=features_df.index
                )
        else:
            # Otherwise use regular DataFrame
            encoded_cat_df = pd.DataFrame(
                encoded_cat_features,
                columns=expected_encoded_cols,
                index=features_df.index
            )
        
        # Drop 'ssl_issuer' column before joining encoded features
        features_df = features_df.drop(columns=['ssl_issuer'], errors='ignore')

        # Join the one-hot encoded features with the original features
        features_df = pd.concat([features_df, encoded_cat_df], axis=1)

        prediction = model.predict(features_df)[0]
        
        # Clear variables to free memory
        del features_df, encoded_cat_df, encoded_cat_features
        
        # Run garbage collection occasionally to prevent memory buildup
        if np.random.random() < 0.1:  # 10% chance of garbage collection
            gc.collect()
            
        return jsonify({"result": "phishing" if prediction == 1 else "safe"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

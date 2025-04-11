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
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load the pre-trained model and one-hot encoder
model = joblib.load('random_forest_model.pkl')
ohe = joblib.load('onehot_encoder.pkl')

# Extract expected one-hot encoded column names
expected_encoded_cols = ohe.get_feature_names_out(['ssl_issuer'])

# URL Cleaning Functions
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
              "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", "l.ead.me",
              "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", "sniply.io", "short.io", "qrco.de"}

# Use a thread pool for concurrent URL expansion requests
executor = ThreadPoolExecutor(max_workers=5)

def is_shortened_url(url):
    parsed = urlparse(url)
    domain_cleaned = parsed.netloc.lower().strip()
    return domain_cleaned in SHORTENERS or bool(re.fullmatch(r"/[A-Za-z0-9_-]{4,10}", parsed.path))

# Cache results for the same URL
@lru_cache(maxsize=100)

def expand_short_url(url):
    if not is_shortened_url(url):
        return url
    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
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

# Feature Extraction Functions
dns_cache = {}

# Cache DNS record lookup
@lru_cache(maxsize=100)

def get_dns_records(domain, record_type):
    cache_key = f"{domain}_{record_type}"
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
            s.settimeout(5)
            s.connect((domain, 443))
            cert = x509.load_der_x509_certificate(s.getpeercert(binary_form=True))
            ssl_info.update({
                "ssl_valid_days": (cert.not_valid_after_utc - cert.not_valid_before_utc).days,
                "ssl_key_size": cert.public_key().key_size if isinstance(cert.public_key(), (rsa.RSAPublicKey, ec.EllipticCurvePublicKey)) else -1,
                "ssl_self_signed": int(cert.issuer == cert.subject),
                "ssl_common_name_mismatch": int(cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value != domain)
            })
    except Exception:
        pass
    return ssl_info

def extract_features(url):
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    domain = f"{domain_info.domain}.{domain_info.suffix}"

    # Add default ssl_issuer in case it is not found
    ssl_issuer = 'Unknown'

    try:
        ssl_info = get_ssl_info(domain)
        ssl_issuer = ssl_info.get('ssl_issuer', 'Unknown')
    except Exception as e:
        print(f"SSL info error for {domain}: {e}")

    # Prepare feature dictionary
    features = {
        "ssl_issuer": ssl_issuer,
        "url_length": len(url),
        "num_subdomains": parsed_url.netloc.count('.'),
        "domain_length": len(domain_info.domain),
        "https": int(parsed_url.scheme == "https"),
        "ip_in_url": int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", parsed_url.netloc))),
        "path_length": len(parsed_url.path),
        "num_special_chars": sum(url.count(ch) for ch in "@?-=_&% ."),
        "num_digits": sum(c.isdigit() for c in url),
        "tld_in_path": int(domain_info.suffix in parsed_url.path),
        "dot_count_in_path": parsed_url.path.count('.'),
        "num_slashes_in_path": parsed_url.path.count('/'),
        "suspicious_keyword": sum(url.lower().count(k) for k in ["login", "verify", "account", "update", "secure", "bank", "confirm", "activate", "signin", "billing"]),
        "num_dns_records": get_dns_records(domain, "A"),
        "mx_record_exists": int(get_dns_records(domain, "MX") > 0),
        "num_ns_records": get_dns_records(domain, "NS"),
        **get_ssl_info(domain)
    }

    return features

FEATURE_ORDER = [
    "url_length", "num_subdomains", "domain_length", "https", "ip_in_url", "path_length",
    "num_special_chars", "num_digits", "tld_in_path", "dot_count_in_path", "num_slashes_in_path",
    "suspicious_keyword", "num_dns_records", "mx_record_exists", "num_ns_records",
    "ssl_valid_days", "ssl_common_name_mismatch", "ssl_self_signed", "ssl_key_size"
]

@app.route('/predict', methods=['POST'])
def predict():
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
    features_df = pd.DataFrame([features_dict])

    # Handle missing 'ssl_issuer' by adding it with a default value
    if 'ssl_issuer' not in features_df.columns:
        features_df['ssl_issuer'] = 'Unknown'
    
    # One-Hot Encoding for 'ssl_issuer'
    encoded_cat_features = ohe.transform(features_df[['ssl_issuer']])
    encoded_cat_df = pd.DataFrame(encoded_cat_features, columns=expected_encoded_cols)
    
    # Drop 'ssl_issuer' column before joining encoded features
    features_df = features_df.drop(columns=['ssl_issuer'], errors='ignore')

    # Join the one-hot encoded features with the original features
    features_df = features_df.join(encoded_cat_df)

    prediction = model.predict(features_df)[0]

    return jsonify({"result": "phishing" if prediction == 1 else "safe"})

# if __name__ == '__main__':
#     app.run(debug=True)
"""
Feature Extraction Script for URLs

This script processes a dataset of URLs and extracts various features useful for 
phishing detection or other classification tasks. It reads the dataset in chunks, 
processes each URL to extract features related to structure, DNS records, and SSL/TLS 
certificates, and appends the extracted features back to a CSV file.

Key functionalities:
- Extracts URL-based features (length, presence of special characters, subdomains, etc.)
- Extracts DNS-related features (A, MX, NS records)
- Extracts SSL/TLS features (issuer, validity, key size, common name mismatch, etc.)
- Uses caching to optimize repeated queries for DNS and SSL data
- Supports checkpointing to resume processing from a specific row
"""

import pandas as pd
import tldextract
import re
import socket
import ssl
import dns.resolver
import os
from urllib.parse import urlparse
from tqdm import tqdm
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa, ec

# File paths
INPUT_FILE = "../dataset/combined_cleaned.csv"
OUTPUT_FILE = "../dataset/combined_with_features.csv"
chunk_size = 100
num_processed = 382700

# Load the combined dataset in chunks
df_combined_iter = pd.read_csv(INPUT_FILE, chunksize=chunk_size, skiprows=range(1, num_processed + 1), header=0)

# Caches for optimization
dns_cache = {}
ssl_cache = {}

# Retrieves the number of DNS records of a given type for the domain
def get_dns_records(domain, record_type):
    cache_key = f"{domain}_{record_type}"
    if cache_key in dns_cache:
        return dns_cache[cache_key]
    try:
        answers = dns.resolver.resolve(domain, record_type)
        dns_cache[cache_key] = len(answers)
        return len(answers)
    except:
        dns_cache[cache_key] = 0
        return 0

# Fetch SSL certificate details
def get_ssl_info(domain):
    ssl_info = {
        "ssl_valid_days": -1,
        "ssl_issuer": "Unknown",
        "ssl_common_name_mismatch": -1,
        "ssl_self_signed": -1,
        "ssl_key_size": -1
    }

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            # Timeout to prevent hanging on unresponsive domains
            s.settimeout(5)
            s.connect((domain, 443))
            cert_bin = s.getpeercert(binary_form=True)
            cert = x509.load_der_x509_certificate(cert_bin)

            # Extract issuer organization (O) field from the certificate
            for attr in cert.issuer:
                if attr.oid.dotted_string == "2.5.4.10":
                    ssl_info["ssl_issuer"] = attr.value
                    break
            
            # Calculate certificate validity duration
            valid_days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
            ssl_info["ssl_valid_days"] = valid_days

            # Determine public key size
            public_key = cert.public_key()
            if isinstance(public_key, rsa.RSAPublicKey):
                ssl_info["ssl_key_size"] = public_key.key_size
            elif isinstance(public_key, ec.EllipticCurvePublicKey):
                ssl_info["ssl_key_size"] = public_key.curve.key_size

            # Check if self-signed (issuer == subject)
            ssl_info["ssl_self_signed"] = int(cert.issuer == cert.subject)

            # Common name mismatch check
            subject_cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
            ssl_info["ssl_common_name_mismatch"] = int(subject_cn != domain)

    except:
        return ssl_info

    return ssl_info

# Function to extract various features from a given URL
def extract_features(url):
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    features = {}
    
    # URL-based features
    features["url_length"] = len(url)
    features["num_subdomains"] = parsed_url.netloc.count('.')
    features["domain_length"] = len(domain_info.domain)
    features["https"] = 1 if parsed_url.scheme == "https" else 0
    features["ip_in_url"] = 1 if re.match(r"^(\\d{1,3}\\.){3}\\d{1,3}$", parsed_url.netloc) else 0
    features["path_length"] = len(parsed_url.path)
    features["num_special_chars"] = sum(map(url.count, "@?-=_&% ."))
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["tld_in_path"] = 1 if domain_info.suffix in parsed_url.path else 0
    features["dot_count_in_path"] = parsed_url.path.count('.')
    features["num_slashes_in_path"] = parsed_url.path.count('/')
    
    # Suspicious keywords detection
    suspicious_keywords = ["login", "verify", "account", "update", "secure", "bank", "confirm", "activate", "signin", "billing"]
    features["suspicious_keyword"] = sum(url.lower().count(keyword) for keyword in suspicious_keywords)

    # DNS-related features
    domain = f"{domain_info.domain}.{domain_info.suffix}"
    
    features["num_dns_records"] = get_dns_records(domain, "A")
    features["mx_record_exists"] = 1 if get_dns_records(domain, "MX") > 0 else 0
    features["num_ns_records"] = get_dns_records(domain, "NS")
    
    # SSL/TLS-related features
    ssl_info = get_ssl_info(domain)
    features.update(ssl_info)
        
    return features

first_write = not os.path.exists(OUTPUT_FILE)

# Process dataset in chunks
for chunk in tqdm(df_combined_iter, desc="Processing Dataset"):
    tqdm.pandas(desc="Extracting Features")
    
    # Extract features
    feature_dicts = chunk["url"].progress_apply(lambda url: extract_features(url))
    
    # Convert feature dictionaries into a DataFrame
    df_features = pd.DataFrame(feature_dicts.tolist())
    
    # Concatenate original chunk with features
    chunk = pd.concat([chunk.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    
    # Append the chunk to the CSV file
    chunk.to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', header=first_write, index=False)
    
    # Ensure subsequent chunks append correctly
    first_write = False

print("✅ Feature extraction completed. Dataset saved.")
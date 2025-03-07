"""
Phishing URL Cleaning Script

This script processes a dataset of phishing URLs by expanding shortened URLs, 
normalizing formatting, and removing duplicates. It operates in chunks to handle 
large datasets efficiently.

Key functionalities:
- Detects and expands shortened URLs using a predefined list and pattern matching.
- Normalizes URLs by:
  - Converting to lowercase.
  - Expanding short URLs when necessary.
  - Removing "www." from domain names.
  - Stripping query parameters and fragments.
- Removes unreachable and malformed URLs.
- Deduplicates URLs before saving.
- Supports batch processing for handling large datasets efficiently.
"""

import pandas as pd
import requests
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
import re
import os

# Expanded list of known URL shorteners
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
    "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", "l.ead.me",
    "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", "sniply.io", "short.io", "qrco.de"
}

# File paths
INPUT_FILE = "../dataset/phishing.csv"
OUTPUT_FILE = "../dataset/cleaned_phishing_v1.csv"

# Detects whether a given URL is likely shortened
def is_shortened_url(url):
    parsed = urlparse(url)
    
    # Check against known shorteners
    domain_cleaned = parsed.netloc.lower().strip()
    if domain_cleaned in SHORTENERS:
        # print(f"Detected as shortener (known list): {url}")
        return True

    # Pattern-based detection: short path (e.g., /abcd) with no file extension
    short_path = bool(re.fullmatch(r"/[A-Za-z0-9_-]{4,10}", parsed.path))
    no_extension = not re.search(r"\.[a-zA-Z]{2,5}$", parsed.path)

    if short_path and no_extension:
        # print(f"Detected as shortener (pattern-based): {url}")
        return True
    
    return False

# Expands shortened URLs by following HTTP redirects
def expand_short_url(url):
    if not is_shortened_url(url):
        # Skip expansion if not a short URL
        return url

    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
        return response.url
    except requests.RequestException:
        # print(f"Removing unreachable short URL: {url}")
        return None

# Cleans and normalizes URLs by expanding short URLs, removing 'www.', and stripping parameters
def clean_url(url):
    # Expand short URL if applicable
    url = expand_short_url(url)
    if url is None:
        return None

    url = url.lower()
    parsed = urlparse(url)

    # Extract clean domain and remove "www."
    netloc = parsed.netloc.replace("www.", "")

    # Remove query parameters and fragments
    path = parsed.path.split("?")[0]

    # Reconstruct cleaned URL
    clean_url = urlunparse((parsed.scheme, netloc, path, "", "", ""))

    return clean_url.rstrip("/")

# Processing settings
chunk_size = 1000
first_write = not os.path.exists(OUTPUT_FILE)

# Process dataset in chunks
for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, names=["url", "label"], header=0):
    print(f"Cleaning {len(chunk)} URLs...")

    # Apply URL cleaning with a progress bar
    chunk["url"] = [clean_url(url) for url in tqdm(chunk["url"], desc="Processing")]

    # Remove None values (invalid URLs)
    chunk = chunk.dropna(subset=["url"])

    # Remove duplicate URLs
    chunk = chunk.drop_duplicates(subset=["url"])

    # Append cleaned data to the CSV file
    chunk.to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', index=False, header=first_write)

    # Ensure subsequent chunks append correctly
    first_write = False

print("✅ Cleaning completed. Dataset saved.")

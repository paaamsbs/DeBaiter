"""
Phishing URL Cleaning - Version 2

This script performs a second round of cleaning on a previously processed phishing URL dataset. 
It further refines the URLs by:
- Expanding shortened URLs and ensuring they are properly resolved.
- Normalizing URLs by removing "www", stripping tracking parameters, and ensuring proper formatting.
- Skipping invalid IPv6 addresses and malformed URLs.
- Removing duplicate entries after processing.

Key functionalities:
- Detects and expands shortened URLs if needed.
- Cleans URLs by:
  - Lowercasing and stripping unnecessary components.
  - Ensuring proper scheme (HTTP/HTTPS).
  - Removing query parameters and fragments.
- Handles large datasets by processing in chunks.
- Ensures final output is free of duplicate URLs.
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
INPUT_FILE = "../dataset/cleaned_phishing_v1.csv"
OUTPUT_FILE = "../dataset/cleaned_phishing_v2.csv"

# Checks if a URL belongs to a known URL shortener service
def is_shortened_url(url):
    parsed = urlparse(url)
    return parsed.netloc.lower().strip() in SHORTENERS

# Expands shortened URLs by following HTTP redirects
def expand_short_url(url):
    if not is_shortened_url(url):
        # Skip expansion if not a short URL
        return url 

    try:
        response = requests.head(url, allow_redirects=True, timeout=3)
        expanded_url = response.url

        # If the expanded URL is still in the SHORTENERS list, discard it
        if is_shortened_url(expanded_url):
            # print(f"Removing unexpanded short URL: {url}")
            return None

        return expanded_url
    except requests.RequestException:
        # print(f"Removing unreachable short URL: {url}")
        return None

# Cleans and normalizes URLs by expanding short URLs, removing 'www.', and stripping parameters
def clean_url(url):
    # Expand short URL if applicable
    url = url.lower().strip()

    # Ensure proper parsing by adding "http://" if missing
    if not url.startswith(("http://", "https://")):
        # Temporary fix for proper parsing
        url = "http://" + url

    try:
        parsed = urlparse(url)

        # Handle invalid IPv6 URLs
        if re.match(r"^\[?[0-9a-fA-F:]+\]?$", parsed.netloc):
            # print(f"Skipping invalid IPv6 URL: {url}")
            return None  # Skip malformed IPv6 addresses

        # Extract clean domain and remove "www."
        netloc = parsed.netloc.replace("www.", "")

        # Remove query parameters and fragments
        path = parsed.path.split("?")[0].split("#")[0].rstrip("/")

        # Reconstruct cleaned URL
        clean_url = urlunparse((parsed.scheme, netloc, path, "", "", ""))

        return clean_url

    except ValueError:
        # Skip malformed URLs
        # print(f"Skipping malformed URL: {url}")
        return None
    
# Processing settings
chunk_size = 1000
first_write = not os.path.exists(OUTPUT_FILE)

# Process dataset in chunks
for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, skiprows=92000, names=["url", "label"], header=0):
    print(f"Cleaning {len(chunk)} URLs...")

    # Apply URL cleaning with a progress bar
    chunk["url"] = [clean_url(url) for url in tqdm(chunk["url"], desc="Processing")]

    # Remove None values (failed or unexpanded short URLs)
    chunk = chunk.dropna(subset=["url"])

    # Append cleaned data to the CSV file
    chunk.to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', index=False, header=first_write)

    # Ensure subsequent chunks append correctly
    first_write = False

# Remove duplicate URLs
print("Removing duplicates from the final output file...")
df = pd.read_csv(OUTPUT_FILE)
df = df.drop_duplicates(subset=["url"])

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)
print("✅ Cleaning completed. Dataset saved.")
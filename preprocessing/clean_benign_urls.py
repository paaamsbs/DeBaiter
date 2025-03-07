"""
Benign URL Cleaning Script

This script processes a dataset of benign URLs to clean and normalize them for further analysis. 
It removes shortened URLs, standardizes formatting, and limits the number of URLs per domain.

Key functionalities:
- Identifies and removes shortened URLs from a predefined list.
- Normalizes URLs by:
  - Converting to lowercase and stripping whitespace.
  - Ensuring proper scheme formatting.
  - Removing "www." from domain names.
  - Stripping tracking parameters and fragments.
- Removes duplicate and malformed URLs.
- Limits each domain to a maximum of 3 URLs.
- Saves the cleaned dataset to a new CSV file.
"""

import pandas as pd
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm

# List of known URL shorteners
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
    "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", "l.ead.me",
    "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", "sniply.io", "short.io", "qrco.de"
}

# File paths
INPUT_FILE = "../dataset/benign.csv"
OUTPUT_FILE = "../dataset/cleaned_benign.csv"

# Enable progress bars for pandas operations
tqdm.pandas()

# Checks if a given URL belongs to a known URL shortener service
def is_shortened_url(url):
    """Check if a URL belongs to a known shortener service."""
    parsed = urlparse(url)
    return parsed.netloc.lower().strip() in SHORTENERS

# Normalizes a given URL
def clean_url(url):
    """Normalize URLs by removing 'www', stripping tracking parameters, and ensuring proper format."""
    url = url.lower().strip()

    # Ensure proper parsing by adding "http://" if missing
    if not url.startswith(("http://", "https://")):
        # Temporary fix for proper parsing
        url = "http://" + url

    parsed = urlparse(url)

    # Extract clean domain and remove "www."
    netloc = parsed.netloc.replace("www.", "")

    # Remove query parameters and fragments
    path = parsed.path.split("?")[0].split("#")[0].rstrip("/")

    # Reconstruct cleaned URL
    clean_url = urlunparse((parsed.scheme, netloc, path, "", "", ""))

    return clean_url

# Process the dataset
print("Loading dataset and removing shortened URLs...")
df = pd.read_csv(INPUT_FILE, names=["url", "label"], header=0)

# Remove shortened URLs using the predefined list
df = df[~df["url"].progress_apply(is_shortened_url)]

# Clean URLs with progress tracking
tqdm.pandas(desc="Cleaning URLs")
df["url"] = df["url"].progress_apply(clean_url)

# Remove None values (malformed URLs)
df = df.dropna(subset=["url"])

# Remove duplicate URLs
df = df.drop_duplicates(subset=["url"])

# Limit each domain to max 3 URLs
df["domain"] = df["url"].apply(lambda x: urlparse(x).netloc)
df = df.groupby("domain").head(3).drop(columns=["domain"])

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)
print("✅ Cleaning completed. Dataset saved.")
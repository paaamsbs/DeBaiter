import pandas as pd
import requests
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
import re
import os

# 🔹 Expanded list of known URL shorteners
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "q-r.to",
    "lnkd.in", "rb.gy", "cutt.ly", "rebrand.ly", "tr.im", "u.to", "x.co", "j.mp", "l.ead.me",
    "clck.ru", "gg.gg", "shorter.me", "shorturl.at", "bl.ink", "sniply.io", "short.io", "qrco.de"
}

# 🔹 File paths
INPUT_FILE = "dataset/phishing.csv"
OUTPUT_FILE = "dataset/cleaned_phishing_v1.csv"

def is_shortened_url(url):
    """Detect if a URL is likely shortened (without expanding)."""
    parsed = urlparse(url)
    
    domain_cleaned = parsed.netloc.lower().strip()
    if domain_cleaned in SHORTENERS:
        print(f"Detected as shortener (known list): {url}")
        return True

    short_path = bool(re.fullmatch(r"/[A-Za-z0-9_-]{4,10}", parsed.path))
    no_extension = not re.search(r"\.[a-zA-Z]{2,5}$", parsed.path)

    if short_path and no_extension:
        print(f"Detected as shortener (pattern-based): {url}")
        return True
    
    return False

def expand_short_url(url):
    """Expand known shortened URLs only if necessary."""
    if not is_shortened_url(url):
        return url  # Skip expansion if not a short URL

    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
        return response.url
    except requests.RequestException:
        print(f"Removing unreachable short URL: {url}")
        return None

def clean_url(url):
    """Clean and normalize URLs by expanding, removing 'www', and stripping tracking parameters."""
    url = expand_short_url(url)
    if url is None:
        return None

    url = url.lower()
    parsed = urlparse(url)

    netloc = parsed.netloc.replace("www.", "")
    path = parsed.path.split("?")[0]

    clean_url = urlunparse((parsed.scheme, netloc, path, "", "", ""))

    return clean_url.rstrip("/")

chunk_size = 1000  # Process in batches
first_write = not os.path.exists(OUTPUT_FILE)  # Append if output exists

for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, names=["url", "label"], header=0):
    print(f"Cleaning {len(chunk)} URLs...")

    chunk["url"] = [clean_url(url) for url in tqdm(chunk["url"], desc="Processing")]

    chunk = chunk.dropna(subset=["url"])
    chunk = chunk.drop_duplicates(subset=["url"])

    chunk.to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False

print(f"✅ Cleaning completed. Results appended to {OUTPUT_FILE}")

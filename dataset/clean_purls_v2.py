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
INPUT_FILE = "dataset/cleaned_phishing_v1.csv"
OUTPUT_FILE = "dataset/cleaned_phishing_v2.csv"

def is_shortened_url(url):
    """Detect if a URL is likely shortened (without expanding)."""
    parsed = urlparse(url)
    return parsed.netloc.lower().strip() in SHORTENERS

def expand_short_url(url):
    """Expand shortened URLs, ensuring the returned URL is different and not still shortened."""
    if not is_shortened_url(url):
        return url  # Skip expansion if not a short URL

    try:
        response = requests.head(url, allow_redirects=True, timeout=3)
        expanded_url = response.url

        # If the expanded URL is still in the SHORTENERS list, discard it
        if is_shortened_url(expanded_url):
            print(f"❌ Removing unexpanded short URL: {url}")
            return None

        return expanded_url
    except requests.RequestException:
        print(f"❌ Removing unreachable short URL: {url}")
        return None

def clean_url(url):
    """Normalize URLs by removing 'www', stripping tracking parameters, and ensuring proper format."""
    url = url.lower().strip()

    # Ensure proper parsing by adding "http://" if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url  # Temporary fix for proper parsing

    try:
        parsed = urlparse(url)

        # Handle invalid IPv6 URLs
        if re.match(r"^\[?[0-9a-fA-F:]+\]?$", parsed.netloc):
            print(f"❌ Skipping invalid IPv6 URL: {url}")
            return None  # Skip malformed IPv6 addresses

        # Extract clean domain and remove "www."
        netloc = parsed.netloc.replace("www.", "")

        # Remove query parameters and fragments
        path = parsed.path.split("?")[0].split("#")[0].rstrip("/")

        # Reconstruct cleaned URL
        clean_url = urlunparse((parsed.scheme, netloc, path, "", "", ""))
        return clean_url

    except ValueError:
        print(f"❌ Skipping malformed URL: {url}")
        return None  # Skip malformed URLs

chunk_size = 1000  # Process in batches
first_write = not os.path.exists(OUTPUT_FILE)  # Append if output exists

# 🔹 Process CSV in chunks
for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, skiprows=92000, names=["url", "label"], header=0):
    print(f"Cleaning {len(chunk)} URLs...")

    chunk["url"] = [clean_url(url) for url in tqdm(chunk["url"], desc="Processing")]

    # Remove None values (failed or unexpanded short URLs)
    chunk = chunk.dropna(subset=["url"])

    # Append cleaned chunk to output file
    chunk.to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False

# 🔹 Remove duplicates from the final output file
print("🔹 Removing duplicates from the final output file...")
df = pd.read_csv(OUTPUT_FILE)
df = df.drop_duplicates(subset=["url"])
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Cleaning completed. Results saved to {OUTPUT_FILE}")
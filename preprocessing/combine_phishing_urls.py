"""
Phishing URLs Aggregation Script

This script consolidates phishing URLs from multiple sources into a single dataset.
It reads URLs from different sources, assigns appropriate labels, and merges them 
into a unified CSV file for further processing.

Sources:
- PhishTank dataset (CSV)
- Kaggle phishing dataset (CSV)
- OpenPhish dataset (TXT)

Key functionalities:
- Reads and processes different file formats (CSV, TXT)
- Ensures all URLs are labeled as "phishing"
- Merges the data while preserving unique entries
- Saves the final dataset to phishing.csv
"""

import pandas as pd

# Load PhishTank dataset, extract the 'url' column, and add 'type' column
df_phishtank = pd.read_csv("../dataset/phishtank_urls.csv", usecols=["url"])
df_phishtank["type"] = "phishing" 

# Load Kaggle phishing dataset (already has 'url' and 'type')
df_kaggle = pd.read_csv("../dataset/kaggle_phishing_urls.csv", usecols=["url", "type"])

# Load OpenPhish dataset (each line is a URL)
with open("../dataset/openphish_urls.txt", "r") as file:
    openphish_urls = [line.strip() for line in file if line.strip()]

# Convert OpenPhish URLs to a DataFrame and label as phishing
df_openphish = pd.DataFrame({"url": openphish_urls})
df_openphish["type"] = "phishing"

# Combine all datasets
df_combined = pd.concat([df_phishtank, df_kaggle, df_openphish], ignore_index=True)

# Save the final phishing dataset
df_combined.to_csv("../dataset/phishing.csv", index=False)

print("✅ Aggregation completed. Dataset saved.")
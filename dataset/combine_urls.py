import pandas as pd

# Read phishtank_urls.csv and extract the 'url' column
df_phishtank = pd.read_csv("phishtank_urls.csv", usecols=["url"])
df_phishtank["type"] = "phishing"  # Add 'type' column

# Read kaggle_phishing_urls.csv (already has 'url' and 'type')
df_kaggle = pd.read_csv("kaggle_phishing_urls.csv", usecols=["url", "type"])

# Read openphish.txt (each line is a URL)
with open("openphish_urls.txt", "r") as file:
    openphish_urls = [line.strip() for line in file if line.strip()]  # Remove empty lines

# Convert OpenPhish URLs to a DataFrame
df_openphish = pd.DataFrame({"url": openphish_urls})
df_openphish["type"] = "phishing"

# Combine all DataFrames
df_combined = pd.concat([df_phishtank, df_kaggle, df_openphish], ignore_index=True)

# Save to phishing.csv
df_combined.to_csv("phishing.csv", index=False)

print("phishing.csv created successfully!")
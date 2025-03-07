"""
Phishing and Benign URLs Merging Script

This script combines cleaned phishing and benign URL datasets into a single labeled dataset.
It assigns labels (1 for phishing, 0 for benign) and merges both datasets for further analysis.

Key functionalities:
- Loads cleaned phishing and benign URL datasets
- Assigns appropriate labels to each category
- Merges the datasets while preserving their integrity
- Saves the final combined dataset as a CSV file
"""

import pandas as pd

# Load datasets
df_phishing = pd.read_csv("../dataset/cleaned_phishing.csv")
df_benign = pd.read_csv("../dataset/cleaned_benign_v2.csv")

# Add labels
df_phishing["label"] = 1
df_benign["label"] = 0

# Concatenate datasets
df_combined = pd.concat([df_phishing, df_benign], ignore_index=True)

# Save the combined dataset
df_combined.to_csv("../dataset/combined_cleaned.csv", index=False)
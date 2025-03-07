"""
Dataset Splitting Script for Model Training

This script splits a cleaned dataset with extracted features into training and testing sets.
It ensures balanced class distribution using stratified sampling and removes any missing labels 
before performing the split. The resulting datasets are saved for model training.

Key functionalities:
- Loads a dataset containing phishing and benign URLs with extracted features
- Removes rows with missing labels to ensure data integrity
- Splits the dataset into 80% training and 20% testing while maintaining class balance
- Saves the resulting train and test sets into separate CSV files
- Prints class distributions to verify stratified sampling effectiveness
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
INPUT_FILE = "../dataset/combined_with_features.csv"
TRAIN_FILE = "../dataset/train.csv"
TEST_FILE = "../dataset/test.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Ensure no missing values in 'label' and remove rows with missing labels
df = df.dropna(subset=["label"])

# Ensure no NaNs remain
df = df.dropna(subset=["label"])  

y = df["label"]
X = df.drop(columns=["label"])

# Check for NaNs before proceeding
print(f"Missing values in y: {y.isna().sum()}")  
print(f"Dataset size: {df.shape[0]} rows")

# Split dataset (80% train, 20% test) while keeping class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# Print class distribution
print("\nClass distribution in Train set:")
print(train_df["label"].value_counts())

print("\nClass distribution in Test set:")
print(test_df["label"].value_counts())

print(f"\n✅ Train set: {X_train.shape[0]} rows")
print(f"✅ Test set: {X_test.shape[0]} rows")
print(f"✅ Data saved to {TRAIN_FILE} and {TEST_FILE}")
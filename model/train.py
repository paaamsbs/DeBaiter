"""
Phishing Detection Model Training and Evaluation Script

This script processes the provided dataset of URLs to train and evaluate various machine learning models aimed at detecting phishing URLs. 
The models include Random Forest, XGBoost, Logistic Regression, and CatBoost. The script also performs data preprocessing, feature extraction, 
model training, evaluation, and the extraction of feature importance metrics.

Key functionalities:
- Data Loading: Reads the training and testing datasets from CSV files.
- Data Preprocessing:
  - Drops the 'url' column (if present) as it's not needed for model training.
  - One-hot encodes categorical features for models that require numerical input.
  - Scales features for models like Logistic Regression that require feature scaling.
- Model Training:
  - Random Forest, XGBoost, and Logistic Regression models are trained using different feature sets (one-hot encoded, scaled, etc.).
  - CatBoost model is trained using categorical features directly.
- Model Evaluation:
  - Accuracy and classification reports for each model are printed.
  - Confusion matrix metrics (True Positive, False Positive, True Negative, False Negative) are computed and displayed in both raw counts and percentages.
- Feature Importance Extraction:
  - The script extracts feature importances from the trained Random Forest model and plots the top 25 important features.
- Model Saving: Saves the trained Random Forest model to a file for future use.

Model Details:
- Random Forest: A Random Forest classifier is trained with 500 estimators, balanced class weights, and a minimum samples split of 5.
- XGBoost: An XGBoost classifier with 500 estimators and a class balance parameter (`scale_pos_weight`) is trained.
- Logistic Regression: A Logistic Regression model is trained with 1000 iterations and balanced class weights.
- CatBoost: A CatBoost classifier with 500 iterations and class weights is trained using categorical features directly.

Output:
- Printed accuracy scores and classification reports for each model.
- Confusion matrix with raw counts and percentage values for each model.
- A bar plot of the top 25 most important features from the trained Random Forest model.
- The trained Random Forest model is saved as `random_forest_model.pkl`.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib

train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv("../dataset/test.csv")

print(train_df.shape)
print(test_df.shape)

# Optionally drop columns that are not needed (e.g., 'url')
if 'url' in train_df.columns:
    train_df = train_df.drop(columns=['url'])
if 'url' in test_df.columns:
    test_df = test_df.drop(columns=['url'])

# Separate features and target
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Identify categorical columns in training data
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
print("Original categorical columns:")
print(cat_cols.tolist())

# Initialize OneHotEncoder with handle_unknown set to 'ignore'
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the training set categorical columns
X_train_encoded = ohe.fit_transform(X_train[cat_cols])
encoded_cols = ohe.get_feature_names_out(cat_cols)
print("\nNew one-hot encoded columns:")
print(encoded_cols.tolist())

# Convert the encoded training array to a DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)

# Create the final training DataFrame by dropping original categorical columns and joining the new ones
X_train_final = X_train.drop(columns=cat_cols).join(X_train_encoded_df)

# Encode the test set using the same encoder
X_test_encoded = ohe.transform(X_test[cat_cols])
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# Create the final test DataFrame by dropping original categorical columns and joining the new ones
X_test_final = X_test.drop(columns=cat_cols).join(X_test_encoded_df)

# Optionally, display a preview of the final encoded DataFrames
print("\nPreview of the training DataFrame after one-hot encoding:")
print(X_train_final.head())

print("\nPreview of the test DataFrame after one-hot encoding:")
print(X_test_final.head())

# Scale features for models that require scaling (Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', max_depth=None, min_samples_split=5)
xgb_model = xgb.XGBClassifier(n_estimators=500, eval_metric='logloss', random_state=42, scale_pos_weight=211017/106367)
logreg_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
cat_model = CatBoostClassifier(iterations=500, learning_rate=0.1, random_seed=42, verbose=0, class_weights=[1, 211017/106367])

# Train models on appropriate data

# For non-scaled one-hot encoded data:
rf_model.fit(X_train_final, y_train)
xgb_model.fit(X_train_final, y_train)

# For non-scaled original data:
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_model.fit(X_train, y_train, cat_features=cat_features)

# For scaled one-hot encoded data:
logreg_model.fit(X_train_scaled, y_train)

# For models trained on one-hot encoded data:
y_pred_rf = rf_model.predict(X_test_final)
y_pred_xgb = xgb_model.predict(X_test_final)

# For CatBoost (trained on original data with categorical features):
y_pred_cat = cat_model.predict(X_test)

# For Logistic Regression (trained on scaled data):
y_pred_logreg = logreg_model.predict(X_test_scaled)

# Evaluate and print metrics for each model

print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("\n---------------------------------\n")

print("XGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("\n---------------------------------\n")

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("\n---------------------------------\n")

print("CatBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))

def print_confusion_metrics(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)

    # Print raw counts
    print(f"{model_name} Confusion Matrix (Counts):")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")

    # Calculate percentages relative to total samples
    tp_percent = (tp / total) * 100
    fp_percent = (fp / total) * 100
    tn_percent = (tn / total) * 100
    fn_percent = (fn / total) * 100

    # Print percentages in the same format
    print(f"{model_name} Confusion Matrix (Percentages):")
    print(f"TP: {tp_percent:.2f}%, FP: {fp_percent:.2f}%, TN: {tn_percent:.2f}%, FN: {fn_percent:.2f}%\n")

# Compute and display confusion metrics for each model

print_confusion_metrics(y_test, y_pred_rf, "Random Forest")
print_confusion_metrics(y_test, y_pred_xgb, "XGBoost")
print_confusion_metrics(y_test, y_pred_logreg, "Logistic Regression")
print_confusion_metrics(y_test, y_pred_cat, "CatBoost")

# Extract feature importances from the trained Random Forest model
importances = rf_model.feature_importances_
features = X_train_final.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort the features by importance in descending order and select the top 10
top_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(25)

# Print the top 10 feature importances
print("Random Forest Feature Importances:")
print(top_importance_df)

# Plot the top 10 feature importances
plt.figure(figsize=(8, 6))
plt.barh(top_importance_df['Feature'], top_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()  # Highest importance appears on top
plt.show()

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved successfully!")
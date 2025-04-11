#!/bin/bash

FILE_ID="1qaQQsPTHvnj2MgNbV45wsh4ZXI3p3nW-"  # <-- Replace with your actual file ID
FILE_NAME="random_forest_model.pkl"

echo "Downloading model from Google Drive..."
gdown --id $FILE_ID --output $FILE_NAME
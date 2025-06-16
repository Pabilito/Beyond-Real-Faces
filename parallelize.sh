#!/bin/bash

FOLDER_NAME=$1

DATASET_DIR="Dataset/$FOLDER_NAME"
ZIP_FILE=$(find "$DATASET_DIR" -maxdepth 1 -name '*.zip')
EXTRACT_DIR="$DATASET_DIR/extracted"
OUTPUT_DIR="Output"

mkdir -p "$EXTRACT_DIR" "$OUTPUT_DIR"

echo "Unzipping $ZIP_FILE..."
unzip -qq "$ZIP_FILE" -d "$EXTRACT_DIR"

echo "Running Python processing..."
python3 Compute_Embeddings_Batch.py "$EXTRACT_DIR" "$OUTPUT_DIR"

echo "Cleaning up $EXTRACT_DIR..."
rm -rf "$EXTRACT_DIR"
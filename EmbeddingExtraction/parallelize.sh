#!/bin/bash
#SBATCH --job-name Embeddings_Computation
#SBATCH -c 16
#SBATCH --time=2-00:00:00
#SBATCH --partition batch
#SBATCH --qos normal

eval "$(micromamba shell hook --shell=bash)"
micromamba activate python3_11_9

FOLDER_NAME=$1
DATASET_DIR="Dataset/$FOLDER_NAME"
OUTPUT_DIR="Output"

mkdir -p "$OUTPUT_DIR"

# Loop through each zip file in the dataset folder (non-recursively)
for ZIP_FILE in "$DATASET_DIR"/*.zip; do
    if [ -f "$ZIP_FILE" ]; then
        ZIP_BASENAME=$(basename "$ZIP_FILE" .zip)
        EXTRACT_DIR="$DATASET_DIR/extracted_$ZIP_BASENAME"
        
        echo "Unzipping $ZIP_FILE to $EXTRACT_DIR..."
        mkdir -p "$EXTRACT_DIR"
        unzip -qq "$ZIP_FILE" -d "$EXTRACT_DIR"

        echo "Running Python processing on $EXTRACT_DIR..."
        python3 Compute_Embeddings_Batch.py "$EXTRACT_DIR" "$OUTPUT_DIR"

        echo "Cleaning up $EXTRACT_DIR..."
        rm -rf "$EXTRACT_DIR"
    fi
done

micromamba deactivate

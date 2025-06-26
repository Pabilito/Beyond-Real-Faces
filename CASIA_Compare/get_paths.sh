#!/bin/bash

# Set your input directory
INPUT_DIR="Closest"

# Create output directory if not exists
OUTPUT_DIR="./Extracted_paths"
mkdir -p "$OUTPUT_DIR"

# Loop through all JSON files
for json_file in "$INPUT_DIR"/*.json; do
    # Extract base name (without extension)
    base_name=$(basename "$json_file" .json)

    # Define output CSV file
    output_csv="$OUTPUT_DIR/${base_name}_matches.csv"

    # Write CSV header
    echo "query_path,casia_path,similarity_score" > "$output_csv"

    # Extract and write data
    jq -r '.results[] | [.query_path, .casia_path, .similarity_score] | @csv' "$json_file" >> "$output_csv"

    echo "Saved: $output_csv"
done
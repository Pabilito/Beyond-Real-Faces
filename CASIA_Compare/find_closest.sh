#!/bin/bash

input_folder="Embeddings"
for file in "$input_folder"/*.json; do
    [[ "$(basename "$file")" == "CASIA.json" ]] && continue
    echo "Processing: $file"
    # Count number of lines in the JSON file
    n_comparisons=$(wc -l < "$file")
    python find_closest.py "$file" --n_comparisons "$n_comparisons" --casia_file "$input_folder"/CASIA.json
done

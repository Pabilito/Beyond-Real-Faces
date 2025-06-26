#!/bin/bash

input_folder="Embeddings"
n_comparisons=10000

for file in "$input_folder"/*.json; do
    [[ "$(basename "$file")" == "CASIA.json" ]] && continue
    echo "Processing: $file"
    python compare_embeddings.py "$file" --n_comparisons "$n_comparisons"
done

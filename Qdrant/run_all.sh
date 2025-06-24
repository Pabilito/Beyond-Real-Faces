#!/bin/bash

input_folder="Embeddings"
n_comparisons=10000

for file in "$input_folder"/*.json; do
    echo "Processing: $file"
    python analyze_embeddings.py "$file" --n_comparisons "$n_comparisons"
done

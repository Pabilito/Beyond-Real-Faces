#!/bin/bash

for folder in Dataset/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo "Submitting job for folder $folder_name"
        sbatch parallelize.sh "$folder_name"
    fi
done
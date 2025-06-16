#!/bin/bash -l
#SBATCH --job-name Embeddings_Computation
#SBATCH -c 16
#SBATCH --time=2-00:00:00
#SBATCH --partition batch
#SBATCH --qos normal

source ~/miniconda3/bin/activate
conda activate python3_11_9

python Compute_Embeddings.py

conda deactivate
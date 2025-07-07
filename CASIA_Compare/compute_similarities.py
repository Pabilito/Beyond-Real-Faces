import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
from pathlib import Path

def load_embeddings(json_file):
    """Load embeddings from JSON file line by line"""
    embeddings = []
    identities = []
    image_names = []
    
    with open(json_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    entry = json.loads(line)
                    embeddings.append(entry['embedding'])
                    identities.append(entry['identity'])
                    image_names.append(entry['image_name'])
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    
    return np.array(embeddings), identities, image_names

def find_best_matches_and_save(input_embeddings, casia_embeddings, n_samples, output_path, batch_size=1000):
    """Find closest CASIA matches for random samples and write results in batches to a file."""
    
    # Ensure the output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Randomly select sample indices
    if len(input_embeddings) < n_samples:
        sample_indices = list(range(len(input_embeddings)))
        print(f"Warning: Only {len(input_embeddings)} embeddings available, using all of them")
    else:
        sample_indices = random.sample(range(len(input_embeddings)), n_samples)

    print(f"Writing best similarities for {len(sample_indices)} samples to {output_path}")
    
    buffer = []

    with open(output_path, 'w') as f:
        for i, sample_idx in enumerate(sample_indices):
            sample_embedding = input_embeddings[sample_idx:sample_idx + 1]
            similarities = cosine_similarity(sample_embedding, casia_embeddings)[0]
            best_sim = np.max(similarities)

            buffer.append(f"{best_sim:.6f}\n")

            # Write in batches
            if len(buffer) >= batch_size:
                f.writelines(buffer)
                buffer = []

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(sample_indices)} samples")

        # Write any remaining values
        if buffer:
            f.writelines(buffer)

    print(f"Similarity scores saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Find best matches in CASIA and save similarity scores')
    parser.add_argument('input_file', help='Input JSON file with embeddings')
    parser.add_argument('--n_comparisons', type=int, default=10000, 
                        help='Number of samples to compare (default: 10000)')
    parser.add_argument('--casia_file', default='CASIA.json',
                        help='CASIA reference file (default: CASIA.json)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    if not os.path.exists(args.casia_file):
        print(f"Error: CASIA file '{args.casia_file}' not found!")
        return
    
    print(f"Loading embeddings from {args.input_file}...")
    input_embeddings, input_identities, input_names = load_embeddings(args.input_file)
    print(f"Loaded {len(input_embeddings)} input embeddings")
    
    print(f"Loading embeddings from {args.casia_file}...")
    casia_embeddings, casia_identities, casia_names = load_embeddings(args.casia_file)
    print(f"Loaded {len(casia_embeddings)} CASIA embeddings")
    
    # Define output path in /SimilarityScores directory
    base_name = Path(args.input_file).stem
    output_dir = Path("SimilarityScores")
    output_path = output_dir / f"{base_name}_BestMatches_{args.n_comparisons}.txt"

    # Find best matches and save scores
    find_best_matches_and_save(input_embeddings, casia_embeddings, args.n_comparisons, output_path)
    
    print(f"\nProcessing complete for {args.input_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()

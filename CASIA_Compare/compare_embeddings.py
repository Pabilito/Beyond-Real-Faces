import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def find_best_matches(input_embeddings, casia_embeddings, n_samples):
    """Find the closest embedding in CASIA for n random samples from input"""
    
    # Randomly sample from input embeddings
    if len(input_embeddings) < n_samples:
        sample_indices = list(range(len(input_embeddings)))
        print(f"Warning: Only {len(input_embeddings)} embeddings available, using all of them")
    else:
        sample_indices = random.sample(range(len(input_embeddings)), n_samples)
    
    best_similarities = []
    
    print(f"Finding best matches for {len(sample_indices)} samples...")
    
    for i, sample_idx in enumerate(sample_indices):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(sample_indices)} samples")
        
        # Get the sample embedding
        sample_embedding = input_embeddings[sample_idx:sample_idx+1]
        
        # Compute similarities with all CASIA embeddings
        similarities = cosine_similarity(sample_embedding, casia_embeddings)[0]
        
        # Find the best (maximum) similarity
        best_sim = np.max(similarities)
        best_similarities.append(best_sim)
    
    return best_similarities

def create_similarity_plot(best_similarities, filename, n_comparisons):
    """Create and save the similarity distribution plot"""
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.histplot(best_similarities, bins=50, kde=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Best Match Similarities\n({len(best_similarities)} samples from {Path(filename).name})")
    plt.xlabel("Cosine Similarity (Best Match)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    mean_sim = np.mean(best_similarities)
    std_sim = np.std(best_similarities)
    plt.axvline(mean_sim, color='red', linestyle='--', alpha=0.8, 
                label=f'Mean: {mean_sim:.3f} ± {std_sim:.3f}')
    plt.legend()
    
    # Generate output filename based on input filename
    base_name = Path(filename).stem
    output_filename = f"{base_name}_BestMatches_{n_comparisons}.png"
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    return output_filename

def main():
    parser = argparse.ArgumentParser(description='Find best matches in CASIA and plot similarity distribution')
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
    
    # Find best matches
    best_similarities = find_best_matches(input_embeddings, casia_embeddings, args.n_comparisons)
    
    # Create and display plot
    output_file = create_similarity_plot(best_similarities, args.input_file, args.n_comparisons)
    
    print(f"\nProcessing complete for {args.input_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
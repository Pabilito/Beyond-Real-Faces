import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import seaborn as sns

def load_embeddings(json_file_path):

    with open(json_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Group embeddings by identity
    identity_groups = defaultdict(list)
    for item in data:
        identity = item['identity']
        embedding = np.array(item['embedding'])
        identity_groups[identity].append({
            'embedding': embedding,
            'image_name': item['image_name'],
            'original_data': item
        })
    
    return identity_groups

def compute_mated_similarities(identity_groups, n_identities=None):

    mated_similarities = []
    identities = list(identity_groups.keys())
    
    if n_identities:
        identities = identities[:n_identities]
    
    for identity in identities:
        embeddings_list = identity_groups[identity]
        
        # Compute all pairwise similarities within the same identity
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                emb1 = embeddings_list[i]['embedding'].reshape(1, -1)
                emb2 = embeddings_list[j]['embedding'].reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                mated_similarities.append(similarity)
    
    return mated_similarities

def compute_non_mated_similarities(identity_groups, n_identities=None, n_comparisons=100):

    non_mated_similarities = []
    identities = list(identity_groups.keys())
    
    if n_identities:
        identities = identities[:n_identities]
    
    # Create a flat list of all embeddings with their identity labels
    all_embeddings = []
    for identity in identities:
        for emb_data in identity_groups[identity]:
            all_embeddings.append({
                'embedding': emb_data['embedding'],
                'identity': identity
            })
    
    # Generate random non-mated pairs
    comparisons_made = 0
    attempts = 0
    max_attempts = n_comparisons * 10  # Prevent infinite loop
    
    while comparisons_made < n_comparisons and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two embeddings
        idx1, idx2 = random.sample(range(len(all_embeddings)), 2)
        emb1_data = all_embeddings[idx1]
        emb2_data = all_embeddings[idx2]
        
        # Only compute if they have different identities
        if emb1_data['identity'] != emb2_data['identity']:
            emb1 = emb1_data['embedding'].reshape(1, -1)
            emb2 = emb2_data['embedding'].reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            non_mated_similarities.append(similarity)
            comparisons_made += 1
    
    if comparisons_made < n_comparisons:
        print(f"Warning: Only generated {comparisons_made} non-mated comparisons out of {n_comparisons} requested")
    
    return non_mated_similarities

def plot_similarity_distributions(mated_sims, non_mated_sims, n_identities, filename):
    """Create KDE plot comparing mated vs non-mated similarity distributions."""
    
    # Check if we have data to plot
    if len(mated_sims) == 0:
        print("Warning: No mated similarities to plot!")
        return None
    if len(non_mated_sims) == 0:
        print("Warning: No non-mated similarities to plot!")
        return None
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.kdeplot(mated_sims, label="Mated (same identity)", fill=True, alpha=0.5)
    sns.kdeplot(non_mated_sims, label="Non-mated (different identity)", fill=True, alpha=0.5)
    plt.title(f"Cosine Similarity Distribution (Sampled {n_identities} identities)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    
    # Generate output filename based on input filename
    base_name = filename.split('.')[0]  # Remove extension
    output_filename = f"{base_name}_MatedVsNonmated.png"
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    plt.show()
    
    return output_filename

def main(json_file_path, n_identities=None, n_non_mated_comparisons=100):

    print("Loading embeddings...")
    identity_groups = load_embeddings(json_file_path)
    
    print(f"Loaded {len(identity_groups)} identities")
    
    # Determine actual number of identities to use
    actual_n_identities = n_identities if n_identities else len(identity_groups)
    if n_identities and n_identities > len(identity_groups):
        actual_n_identities = len(identity_groups)
        print(f"Requested {n_identities} identities, but only {len(identity_groups)} available")
    
    print(f"Using {actual_n_identities} identities for analysis")
    
    print("Computing mated similarities...")
    mated_similarities = compute_mated_similarities(identity_groups, n_identities)
    
    print("Computing non-mated similarities...")
    non_mated_similarities = compute_non_mated_similarities(
        identity_groups, n_identities, n_non_mated_comparisons
    )
    
    print(f"Generated {len(mated_similarities)} mated comparisons")
    print(f"Generated {len(non_mated_similarities)} non-mated comparisons")
    
    # Check if we have enough data
    if len(mated_similarities) == 0:
        print("Error: No mated similarities found. Check if identities have multiple samples.")
        return None, None, identity_groups
    
    if len(non_mated_similarities) == 0:
        print("Error: No non-mated similarities found. Check if you have multiple identities.")
        return None, None, identity_groups
    
    # Create visualization
    print("Creating visualization...")
    import os
    filename = os.path.basename(json_file_path) + str(n_identities)
    filename = os.path.join("Figures", filename)
    plot_similarity_distributions(mated_similarities, non_mated_similarities, actual_n_identities, filename)
    
    return mated_similarities, non_mated_similarities, identity_groups


if __name__ == "__main__":

    json_file_path = "Embeddings/Syn-Multi-PIE.json"
    
    mated_sims, non_mated_sims, groups = main(
        json_file_path, 
        n_identities=10,
        n_non_mated_comparisons=100
    )
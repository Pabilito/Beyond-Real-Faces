import json
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
import argparse

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

def compute_limited_mated_similarities(identity_groups, n_comparisons=1000):
    mated_similarities = []
    identities = list(identity_groups.keys())
    
    # Generate all possible mated pairs
    all_mated_pairs = []
    for identity in identities:
        embeddings_list = identity_groups[identity]
        
        # Generate all pairwise combinations within the same identity
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                all_mated_pairs.append({
                    'identity': identity,
                    'emb1': embeddings_list[i]['embedding'],
                    'emb2': embeddings_list[j]['embedding']
                })
    
    print(f"Total possible mated pairs: {len(all_mated_pairs)}")
    
    # Randomly sample n_comparisons pairs
    if len(all_mated_pairs) > n_comparisons:
        selected_pairs = random.sample(all_mated_pairs, n_comparisons)
    else:
        selected_pairs = all_mated_pairs
        print(f"Warning: Only {len(all_mated_pairs)} mated pairs available, using all of them")
    
    # Compute similarities for selected pairs
    for pair in selected_pairs:
        emb1 = pair['emb1'].reshape(1, -1)
        emb2 = pair['emb2'].reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        mated_similarities.append(similarity)
    
    return mated_similarities

def compute_limited_non_mated_similarities(identity_groups, n_comparisons=1000):
    non_mated_similarities = []
    identities = list(identity_groups.keys())
    
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

def save_similarities_to_file(mated_sims, non_mated_sims, filename_prefix):
    os.makedirs("SimilarityScores", exist_ok=True)
    
    mated_file = os.path.join("SimilarityScores", f"{filename_prefix}_mated_similarities.txt")
    non_mated_file = os.path.join("SimilarityScores", f"{filename_prefix}_non_mated_similarities.txt")
    
    # Save mated similarities, one per line
    with open(mated_file, 'w') as f:
        for sim in mated_sims:
            f.write(f"{sim}\n")
    
    # Save non-mated similarities, one per line
    with open(non_mated_file, 'w') as f:
        for sim in non_mated_sims:
            f.write(f"{sim}\n")
    
    print(f"Mated similarities saved to: {mated_file}")
    print(f"Non-mated similarities saved to: {non_mated_file}")

def main(json_file_path, n_comparisons=1000):
    print("Loading embeddings...")
    identity_groups = load_embeddings(json_file_path)
    
    print(f"Loaded {len(identity_groups)} identities")
    
    # Count total samples
    total_samples = sum(len(embs) for embs in identity_groups.values())
    print(f"Total samples: {total_samples}")
    
    print(f"Computing {n_comparisons} mated similarities...")
    mated_similarities = compute_limited_mated_similarities(identity_groups, n_comparisons)
    
    print(f"Computing {n_comparisons} non-mated similarities...")
    non_mated_similarities = compute_limited_non_mated_similarities(identity_groups, n_comparisons)
    
    print(f"Generated {len(mated_similarities)} mated comparisons")
    print(f"Generated {len(non_mated_similarities)} non-mated comparisons")
    
    if len(mated_similarities) == 0:
        print("Error: No mated similarities found. Check if identities have multiple samples.")
        return None, None, identity_groups
    
    if len(non_mated_similarities) == 0:
        print("Error: No non-mated similarities found. Check if you have multiple identities.")
        return None, None, identity_groups
    
    print("Saving similarity scores to files...")
    filename_prefix = os.path.basename(json_file_path).split('.')[0]  # base name without extension
    save_similarities_to_file(mated_similarities, non_mated_similarities, filename_prefix)
    
    return mated_similarities, non_mated_similarities, identity_groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mated and non-mated similarity scores.")
    parser.add_argument("json_file_path", type=str, help="Path to the input JSON file.")
    parser.add_argument("--n_comparisons", type=int, default=10000, help="Number of comparisons to compute (default: 10000)")
    args = parser.parse_args()

    mated_sims, non_mated_sims, groups = main(
        args.json_file_path,
        n_comparisons=args.n_comparisons
    )

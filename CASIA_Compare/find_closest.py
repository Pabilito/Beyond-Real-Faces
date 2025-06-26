import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
from pathlib import Path
from typing import List, Tuple, Generator
import gc

def load_embeddings_generator(json_file: str) -> Generator[Tuple[np.ndarray, str, str], None, None]:
    """Generator to load embeddings one by one to save memory"""
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    embedding = np.array(entry['embedding'], dtype=np.float32)
                    yield embedding, entry['identity'], entry['image_name']
                except (json.JSONDecodeError, KeyError):
                    continue

def count_lines(json_file: str) -> int:
    """Count valid lines in JSON file"""
    count = 0
    with open(json_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    continue
    return count

def load_embeddings_batch(json_file: str, batch_size: int = 50000) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load embeddings in batches with memory optimization"""
    print(f"Counting lines in {json_file}...")
    total_lines = count_lines(json_file)
    print(f"Found {total_lines} valid entries")
    
    # Pre-allocate arrays for better memory efficiency
    embeddings_list = []
    identities = []
    image_names = []
    
    batch_count = 0
    for embedding, identity, image_name in load_embeddings_generator(json_file):
        embeddings_list.append(embedding)
        identities.append(identity)
        image_names.append(image_name)
        
        batch_count += 1
        if batch_count % batch_size == 0:
            print(f"  Loaded {batch_count}/{total_lines} entries")
    
    # Convert to numpy array with optimal dtype
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    del embeddings_list  # Free memory
    gc.collect()
    
    return embeddings, identities, image_names

def find_top_matches_optimized(input_embeddings: np.ndarray, input_identities: List[str], 
                              input_names: List[str], casia_embeddings: np.ndarray, 
                              casia_identities: List[str], casia_names: List[str], 
                              n_samples: int, top_k: int = 10, batch_size: int = 1000) -> List[dict]:
    """Optimized version using batch processing and direct dot product (embeddings assumed normalized)"""
    
    # Randomly sample from input embeddings
    total_input = len(input_embeddings)
    if total_input < n_samples:
        sample_indices = list(range(total_input))
        print(f"Warning: Only {total_input} embeddings available, using all of them")
    else:
        sample_indices = random.sample(range(total_input), n_samples)
    
    print("Using pre-normalized embeddings for direct dot product similarity...")
    
    top_matches = []
    total_samples = len(sample_indices)
    
    print(f"Finding top {top_k} matches for {total_samples} samples using batch processing...")
    
    # Process in batches to manage memory
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1} "
              f"({batch_start+1}-{batch_end}/{total_samples})")
        
        # Get batch of query embeddings (already normalized)
        batch_embeddings = input_embeddings[batch_indices]
        
        # Compute similarities for entire batch at once using direct dot product
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(batch_embeddings, casia_embeddings.T)
        
        # Process each sample in the batch
        for i, sample_idx in enumerate(batch_indices):
            # Get similarities for this sample
            sample_similarities = similarities[i]
            
            # Find top K matches using argpartition (faster than argsort for top-k)
            if top_k < len(sample_similarities):
                top_indices = np.argpartition(sample_similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(sample_similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(sample_similarities)[::-1][:top_k]
            
            # Store the results
            sample_matches = {
                "query_identity": input_identities[sample_idx],
                "query_image_name": input_names[sample_idx],
                "query_path": f"{input_identities[sample_idx]}/{input_names[sample_idx]}",
                "matches": []
            }
            
            for rank, casia_idx in enumerate(top_indices):
                match_info = {
                    "rank": rank + 1,
                    "similarity_score": float(sample_similarities[casia_idx]),
                    "casia_identity": casia_identities[casia_idx],
                    "casia_image_name": casia_names[casia_idx],
                    "casia_path": f"{casia_identities[casia_idx]}/{casia_names[casia_idx]}"
                }
                sample_matches["matches"].append(match_info)
            
            top_matches.append(sample_matches)
        
        # Force garbage collection after each batch
        gc.collect()
    
    return top_matches

def save_top_matches_streaming(top_matches: List[dict], input_filename: str, 
                              n_comparisons: int, top_k: int) -> str:
    """Save results using streaming to handle large outputs"""
    
    base_name = Path(input_filename).stem
    output_filename = f"{base_name}_Top{top_k}Matches_{n_comparisons}.json"
    
    print(f"Saving {len(top_matches)} results to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        # Write metadata
        f.write('{\n')
        f.write('  "metadata": {\n')
        f.write(f'    "input_file": "{input_filename}",\n')
        f.write(f'    "n_samples_compared": {len(top_matches)},\n')
        f.write(f'    "top_k_matches": {top_k},\n')
        f.write(f'    "total_comparisons_requested": {n_comparisons}\n')
        f.write('  },\n')
        f.write('  "results": [\n')
        
        # Write results one by one
        for i, match in enumerate(top_matches):
            json.dump(match, f, separators=(',', ':'))
            if i < len(top_matches) - 1:
                f.write(',')
            f.write('\n')
            
            # Progress indicator for large files
            if (i + 1) % 1000 == 0:
                print(f"    Saved {i + 1}/{len(top_matches)} results")
        
        f.write('  ]\n')
        f.write('}\n')
    
    print(f"Results saved to: {output_filename}")
    return output_filename

def main():
    parser = argparse.ArgumentParser(description='Find top K matches in CASIA (optimized for large datasets)')
    parser.add_argument('input_file', help='Input JSON file with embeddings')
    parser.add_argument('--n_comparisons', type=int, default=10000, 
                        help='Number of samples to compare (default: 10000)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top matches to save per sample (default: 10)')
    parser.add_argument('--casia_file', default='CASIA.json',
                        help='CASIA reference file (default: CASIA.json)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing (default: 1000)')
    parser.add_argument('--load_batch_size', type=int, default=50000,
                        help='Batch size for loading embeddings (default: 50000)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    if not os.path.exists(args.casia_file):
        print(f"Error: CASIA file '{args.casia_file}' not found!")
        return
    
    print(f"Loading embeddings from {args.input_file}...")
    input_embeddings, input_identities, input_names = load_embeddings_batch(
        args.input_file, args.load_batch_size)
    print(f"Loaded {len(input_embeddings)} input embeddings")
    
    print(f"Loading embeddings from {args.casia_file}...")
    casia_embeddings, casia_identities, casia_names = load_embeddings_batch(
        args.casia_file, args.load_batch_size)
    print(f"Loaded {len(casia_embeddings)} CASIA embeddings")
    
    # Find top matches with optimized algorithm
    top_matches = find_top_matches_optimized(
        input_embeddings, input_identities, input_names,
        casia_embeddings, casia_identities, casia_names,
        args.n_comparisons, args.top_k, args.batch_size)
    
    # Save results with streaming
    output_file = save_top_matches_streaming(top_matches, args.input_file, 
                                           args.n_comparisons, args.top_k)
    
    print(f"\nProcessing complete for {args.input_file}")
    print(f"Found top {args.top_k} matches for {len(top_matches)} samples")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
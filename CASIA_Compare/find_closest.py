import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
from pathlib import Path
from typing import List, Tuple, Generator
import gc
import heapq

#Avoid problem with negative values on heap
class PositiveTopK:
    def __init__(self, k: int):
        self.k = k
        self._heap = []  # min-heap of (similarity, query_idx, casia_idx)
        self._min_positive = 0.0  # Only consider similarities above this threshold
    
    def add(self, similarity: float, query_idx: int, casia_idx: int):
        """Add similarity only if it's positive and potentially in top K"""
        # Skip negative similarities entirely
        if similarity <= 0:
            return
            
        if len(self._heap) < self.k:
            # Heap not full, add any positive similarity
            heapq.heappush(self._heap, (similarity, query_idx, casia_idx))
            if len(self._heap) == self.k:
                # Just filled up, set minimum threshold
                self._min_positive = self._heap[0][0]
        elif similarity > self._heap[0][0]:  # Better than worst in heap
            # Replace worst with this better one
            heapq.heapreplace(self._heap, (similarity, query_idx, casia_idx))
            self._min_positive = self._heap[0][0]  # Update threshold
    
    def get_sorted_results(self) -> List[Tuple[float, int, int]]:
        """Return all results sorted by similarity (highest first)"""
        return sorted(self._heap, key=lambda x: x[0], reverse=True)
    
    def get_min_positive(self) -> float:
        """Get the minimum positive similarity currently in the top K"""
        return self._min_positive
    
    def get_count(self) -> int:
        return len(self._heap)
    
    def is_full(self) -> bool:
        return len(self._heap) >= self.k

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

def find_top_global_matches(input_embeddings: np.ndarray, input_identities: List[str], 
                           input_names: List[str], casia_embeddings: np.ndarray, 
                           casia_identities: List[str], casia_names: List[str], 
                           n_samples: int, top_global: int = 100, batch_size: int = 1000) -> List[dict]:
    """Find top N highest similarity pairs - positive similarities only"""
    
    # Randomly sample from input embeddings
    total_input = len(input_embeddings)
    if total_input < n_samples:
        sample_indices = list(range(total_input))
        print(f"Warning: Only {total_input} embeddings available, using all of them")
    else:
        sample_indices = random.sample(range(total_input), n_samples)
    
    print("Using pre-normalized embeddings for direct dot product similarity...")
    print("Filtering for positive similarities only...")
    
    # Simple positive-only tracker
    top_k = PositiveTopK(top_global)
    
    total_samples = len(sample_indices)
    print(f"Finding top {top_global} highest positive similarities from {total_samples} samples...")
    
    positive_count = 0
    total_comparisons = 0
    
    # Process in batches to manage memory
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1} "
              f"({batch_start+1}-{batch_end}/{total_samples})")
        
        # Get batch of query embeddings (already normalized)
        batch_embeddings = input_embeddings[batch_indices]
        
        # Compute similarities for entire batch at once
        similarities = np.dot(batch_embeddings, casia_embeddings.T)
        
        # Process each sample in the batch
        for i, sample_idx in enumerate(batch_indices):
            sample_similarities = similarities[i]
            
            # Only process positive similarities
            positive_mask = sample_similarities > 0
            positive_indices = np.where(positive_mask)[0]
            positive_sims = sample_similarities[positive_mask]
            
            positive_count += len(positive_sims)
            total_comparisons += len(sample_similarities)
            
            # Add positive similarities to tracker
            for j, casia_idx in enumerate(positive_indices):
                similarity = positive_sims[j]
                top_k.add(float(similarity), sample_idx, int(casia_idx))
        
        # Progress update
        if top_k.is_full():
            print(f"    Found {top_k.get_count()} positive similarities, "
                  f"current threshold: {top_k.get_min_positive():.6f}")
        else:
            print(f"    Found {top_k.get_count()}/{top_global} positive similarities so far")
        
        # Force garbage collection after each batch
        gc.collect()
    
    print(f"\nStatistics:")
    print(f"  Total comparisons: {total_comparisons:,}")
    print(f"  Positive similarities: {positive_count:,} ({100*positive_count/total_comparisons:.1f}%)")
    print(f"  Final top-{top_global} threshold: {top_k.get_min_positive():.6f}")
    
    # Get final sorted results
    sorted_results = top_k.get_sorted_results()
    
    if len(sorted_results) < top_global:
        print(f"Warning: Only found {len(sorted_results)} positive similarities, less than requested {top_global}")
    
    print(f"Creating final results for top {len(sorted_results)} matches...")
    
    final_matches = []
    for rank, (similarity, query_idx, casia_idx) in enumerate(sorted_results):
        match_info = {
            "rank": rank + 1,
            "similarity_score": similarity,
            "query_identity": input_identities[query_idx],
            "query_image_name": input_names[query_idx],
            "query_path": f"{input_identities[query_idx]}/{input_names[query_idx]}",
            "casia_identity": casia_identities[casia_idx],
            "casia_image_name": casia_names[casia_idx],
            "casia_path": f"{casia_identities[casia_idx]}/{casia_names[casia_idx]}"
        }
        final_matches.append(match_info)
    
    return final_matches

def save_top_matches_streaming(top_matches: List[dict], input_filename: str, 
                              n_comparisons: int, top_global: int) -> str:
    """Save results using streaming to handle large outputs"""
    
    base_name = Path(input_filename).stem
    output_filename = f"{base_name}_Top{top_global}Global_{n_comparisons}.json"
    
    print(f"Saving {len(top_matches)} results to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        # Write metadata
        f.write('{\n')
        f.write('  "metadata": {\n')
        f.write(f'    "input_file": "{input_filename}",\n')
        f.write(f'    "n_samples_compared": {n_comparisons},\n')
        f.write(f'    "top_global_matches": {top_global},\n')
        f.write(f'    "total_pairs_considered": {n_comparisons * len(top_matches)}\n')
        f.write('  },\n')
        f.write('  "results": [\n')
        
        # Write results one by one
        for i, match in enumerate(top_matches):
            json.dump(match, f, separators=(',', ':'))
            if i < len(top_matches) - 1:
                f.write(',')
            f.write('\n')
        
        f.write('  ]\n')
        f.write('}\n')
    
    print(f"Results saved to: {output_filename}")
    return output_filename

def main():
    parser = argparse.ArgumentParser(description='Find top N highest similarity pairs globally')
    parser.add_argument('input_file', help='Input JSON file with embeddings')
    parser.add_argument('--n_comparisons', type=int, default=10000, 
                        help='Number of samples to compare (default: 10000)')
    parser.add_argument('--top_global', type=int, default=100,
                        help='Number of highest similarity pairs to save globally (default: 100)')
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
    
    # Find top global matches
    top_matches = find_top_global_matches(
        input_embeddings, input_identities, input_names,
        casia_embeddings, casia_identities, casia_names,
        args.n_comparisons, args.top_global, args.batch_size)
    
    # Save results with streaming
    output_file = save_top_matches_streaming(top_matches, args.input_file, 
                                           args.n_comparisons, args.top_global)
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
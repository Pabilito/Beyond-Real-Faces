import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "embeddings"

def get_all_identities():
    identities = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_payload=True
        )
        for p in results:
            identities.add(p.payload.get('identity'))
        if offset is None:
            break
    return list(identities)

def get_vectors_by_identity(identity_value, exclude_point_id=None):
    vectors = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_vectors=True,
            with_payload=True
        )
        for p in results:
            if p.payload.get('identity') == identity_value:
                if exclude_point_id is None or p.id != exclude_point_id:
                    vectors.append(p.vector)
        if offset is None:
            break
    return np.array(vectors)

def get_vectors_for_identities(identity_list):
    vectors = []
    offset = None
    identity_set = set(identity_list)
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_vectors=True,
            with_payload=True
        )
        for p in results:
            if p.payload.get('identity') in identity_set:
                vectors.append(p.vector)
        if offset is None:
            break
    return np.array(vectors)

def get_one_point_by_identity(identity_value):
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="identity",
                    match=MatchValue(value=identity_value)
                )
            ]
        ),
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    points = results[0]
    if points:
        return points[0]
    return None

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)

def main(sample_non_mated=100, max_identities=100):
    identities = get_all_identities()
    print(f"Found {len(identities)} unique identities.")
    
    # Randomly sample max_identities identities
    sampled_identities = random.sample(identities, min(max_identities, len(identities)))
    
    all_mated_similarities = []
    all_non_mated_similarities = []
    
    for identity in tqdm(sampled_identities, desc="Processing identities"):
        point = get_one_point_by_identity(identity)
        if point is None:
            continue
        
        ref_vector = np.array(point.vector)
        mated_vectors = get_vectors_by_identity(identity, exclude_point_id=point.id)
        if len(mated_vectors) == 0:
            continue
        
        other_identities = [i for i in sampled_identities if i != identity]
        sampled_non_mated_identities = random.sample(other_identities, min(sample_non_mated, len(other_identities)))
        
        non_mated_vectors = get_vectors_for_identities(sampled_non_mated_identities)
        if len(non_mated_vectors) == 0:
            continue
        
        mated_sim = cosine_similarity(ref_vector, mated_vectors)
        non_mated_sim = cosine_similarity(ref_vector, non_mated_vectors)
        
        all_mated_similarities.extend(mated_sim)
        all_non_mated_similarities.extend(non_mated_sim)
    
    all_mated_similarities = np.array(all_mated_similarities)
    all_non_mated_similarities = np.array(all_non_mated_similarities)
    
    plt.figure(figsize=(12,7))
    sns.kdeplot(all_mated_similarities, label="Mated (same identity)", fill=True, alpha=0.5)
    sns.kdeplot(all_non_mated_similarities, label="Non-mated (different identity)", fill=True, alpha=0.5)
    plt.title(f"Aggregated Cosine Similarity Distribution (Sampled {len(sampled_identities)} identities)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

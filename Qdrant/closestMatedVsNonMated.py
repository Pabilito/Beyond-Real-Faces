import matplotlib.pyplot as plt
import seaborn as sns
from qdrant_client import QdrantClient
from tqdm import tqdm  # <- added tqdm

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "embeddings"

def main():
    closest_mated_scores = []
    closest_nonmated_scores = []

    print("Retrieving all points from collection...")
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_vectors=True,
        with_payload=True,
        limit=10000
    )
    print(f"Found {len(points)} points.")

    # Use tqdm progress bar
    for point in tqdm(points, desc="Processing points"):
        point_id = point.id
        vector = point.vector
        identity = point.payload.get("identity")

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=20,
            with_payload=True
        )

        filtered_results = [r for r in results if r.id != point_id]

        mated_scores = [r.score for r in filtered_results if r.payload.get("identity") == identity]
        nonmated_scores = [r.score for r in filtered_results if r.payload.get("identity") != identity]

        if mated_scores:
            closest_mated_scores.append(max(mated_scores))
        if nonmated_scores:
            closest_nonmated_scores.append(max(nonmated_scores))

    print(f"Collected {len(closest_mated_scores)} mated and {len(closest_nonmated_scores)} non-mated similarity scores.")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(closest_mated_scores, label="Closest Mated", shade=True)
    sns.kdeplot(closest_nonmated_scores, label="Closest Non-Mated", shade=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Distribution of Closest Cosine Similarities")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
from qdrant_client import QdrantClient

# Connect to local instance
client = QdrantClient(host="localhost", port=6333)

# Collection name
COLLECTION_NAME = "embeddings"

# Get the vector and payload for point ID 1
point = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1],
    with_vectors=True,
    with_payload=True
)

# Extract the query vector and identity
query_vector = point[0].vector
query_identity = point[0].payload.get('identity')

print(f"Query point identity: {query_identity}")

# Search for similar vectors with filter for same identity
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="identity",
                match=MatchValue(value=query_identity)
            )
        ]
    ),
    limit=5,
    with_payload=True
)

# Print results
print(f"Points with same identity '{query_identity}':")
for i, r in enumerate(results, 1):
    print(f"#{i} - ID: {r.id}, Score: {r.score:.4f}, Payload: {r.payload}")
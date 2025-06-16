import numpy as np
from qdrant_client import QdrantClient

# Connect to local instance
client = QdrantClient(host="localhost", port=6333)

# Collection setup (adjust vector size)
VECTOR_SIZE = 512
COLLECTION_NAME = "my_vectors"

query_vector = np.random.rand(VECTOR_SIZE).astype(np.float32)

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5
)

for r in results:
    print(f"ID: {r.id}, Score: {r.score}, Payload: {r.payload}")
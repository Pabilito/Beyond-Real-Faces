import numpy as np
from qdrant_client import QdrantClient

# Connect to local instance
client = QdrantClient(host="localhost", port=6333)

# Collection name
COLLECTION_NAME = "embeddings"

# Get the vector for point ID 1
point = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1],
    with_vectors=True
)

# Extract the query vector
query_vector = point[0].vector

# Search for similar vectors
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=11,
    with_payload=True
)

# Print results (skip the first one as it's the query point itself)
print("5 Nearest neighbors for point ID 1:")
for i, r in enumerate(results[1:11], 1):  # Skip first result, take next 5
    print(f"#{i} - ID: {r.id}, Score: {r.score:.4f}, Payload: {r.payload}")


'''
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
'''
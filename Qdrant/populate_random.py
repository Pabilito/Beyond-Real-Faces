from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from tqdm import tqdm

'''
docker run -p 6333:6333 qdrant/qdrant
'''


# Connect to local instance
client = QdrantClient(host="localhost", port=6333)

# Collection setup (adjust vector size)
VECTOR_SIZE = 512
COLLECTION_NAME = "my_vectors"

# Create/recreate collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# Generate dummy vectors (replace with your actual vectors)
NUM_VECTORS = 1_000_000
BATCH_SIZE = 1000

for start in tqdm(range(0, NUM_VECTORS, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, NUM_VECTORS)
    vectors = np.random.rand(end - start, VECTOR_SIZE).astype(np.float32)
    
    points = [
        PointStruct(id=int(i), vector=vectors[i - start], payload={"label": f"vec_{i}"})
        for i in range(start, end)
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)

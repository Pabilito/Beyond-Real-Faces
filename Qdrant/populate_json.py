import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

# Connect to local Qdrant instance
client = QdrantClient(host="localhost", port=6333)

# Parameters
COLLECTION_NAME = "embeddings"
VECTOR_SIZE = 512
BATCH_SIZE = 1000
FILE = "Embeddings/DCFace/embeddings.json"

# Load embeddings from JSON file (line-separated JSON)
data = []
with open(FILE, "r") as f:
    for line_num, line in enumerate(f, 1):  # Start line numbering from 1
        line = line.strip()
        if line:  # Skip empty lines
            try:
                item = json.loads(line)
                # Add line number as id and keep original data
                item_with_id = {
                    "id": line_num,
                    "identity": item.get("identity"),
                    "image_name": item.get("image_name"),
                    "embedding": item.get("embedding")
                }
                data.append(item_with_id)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

# Check and infer vector size
if len(data) == 0:
    raise ValueError("No embeddings found in the JSON file.")

VECTOR_SIZE = len(data[0]["embedding"])

# Recreate the collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# Upload in batches
for start in tqdm(range(0, len(data), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(data))
    batch = data[start:end]
    points = [
        PointStruct(
            id=item["id"],
            vector=item["embedding"],
            payload={k: v for k, v in item.items() if k not in ["id", "embedding"]}
        )
        for item in batch
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"Successfully uploaded {len(data)} embeddings to collection '{COLLECTION_NAME}'")
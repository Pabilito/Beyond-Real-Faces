import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "embeddings"

# Example 1: Vector similarity search (returns identity and image_name in results)
def search_similar_vectors(query_vector, limit=5):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        with_payload=True  # This includes the payload (identity, image_name) in results
    )
    
    for result in results:
        print(f"ID: {result.id}")
        print(f"Score: {result.score}")
        print(f"Identity: {result.payload.get('identity')}")
        print(f"Image Name: {result.payload.get('image_name')}")
        print("---")
    
    return results

# Example 2: Filter by identity
def search_by_identity(identity_value):
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
        with_payload=True,
        with_vectors=True  # Include vectors if you need them
    )
    
    return results[0]  # Returns points, next_page_offset

# Example 3: Filter by image name pattern
def search_by_image_name(image_name):
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="image_name",
                    match=MatchValue(value=image_name)
                )
            ]
        ),
        with_payload=True
    )
    
    return results[0]

# Example 4: Get specific point by ID
def get_point_by_id(point_id):
    result = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=True
    )
    
    if result:
        point = result[0]
        print(f"ID: {point.id}")
        print(f"Identity: {point.payload.get('identity')}")
        print(f"Image Name: {point.payload.get('image_name')}")
        print(f"Vector: {point.vector[:5]}...")  # Show first 5 dimensions
    
    return result

# Example 5: Get all unique identities
def get_all_identities():
    # Scroll through all points and collect unique identities
    identities = set()
    offset = None
    
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_payload=True
        )
        
        for point in results:
            if 'identity' in point.payload:
                identities.add(point.payload['identity'])
        
        if offset is None:
            break
    
    return list(identities)

# Example usage:
if __name__ == "__main__":
    # Search for points with specific identity
    identity_results = search_by_identity("5290")
    print(f"Found {len(identity_results)} points with identity '5290'")
    
    # Search for specific image
    image_results = search_by_image_name("52.jpg")
    print(f"Found {len(image_results)} points with image '52.jpg'")
    
    # Get point by line number (ID)
    point = get_point_by_id(1)  # Get first line
    
    # Get all unique identities in the collection
    all_identities = get_all_identities()
    print(f"Total unique identities: {len(all_identities)}")
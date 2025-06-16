"""
To be run with:
* parallelize.sh
* main_runner.sh

Usage:
./main_runner.sh
"""

import torch
import torch.nn.functional as F
import os
import json
from PIL import Image
import iresnet
from torchvision import transforms
import numpy as np
import cv2
import sys

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
FOLDER_PATH = "Models"
MODEL_NAME = "ArcFace_R100_MS1MV3.pth"
weights = os.path.join(FOLDER_PATH, MODEL_NAME)

# Load and configure model
model = iresnet.iresnet100()
model.load_state_dict(torch.load(weights, map_location=DEVICE), strict=True)
model.to(DEVICE)
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(image_path):
    """Load and preprocess image for the model"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_embedding(image_tensor):
    """Extract features from image using the model"""
    if image_tensor is None:
        return None
   
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            features = model(image_tensor)
            # Normalize features for cosine similarity
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_dataset(dataset_path, output_path):
    """Process dataset and write embeddings incrementally to JSON"""
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return

    for identity_folder in os.listdir(dataset_path):
        identity_path = os.path.join(dataset_path, identity_folder)

        if os.path.isdir(identity_path):
            print(f"Processing identity: {identity_folder}")

            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_tensor = load_image(file_path)
                    embedding = get_embedding(image_tensor)
                    embedding = np.round(embedding, 5)

                    if embedding is not None:
                        entry = {
                            "identity": identity_folder,
                            "image_name": file,
                            "embedding": embedding.tolist()
                        }

                        try:
                            with open(output_path, 'a') as f:
                                f.write(json.dumps(entry) + '\n')
                        except Exception as e:
                            print(f"Error writing {file} to JSON: {e}")
                    else:
                        print(f"Failed to process {file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python Compute_Embeddings_Batch.py <dataset_path>")
        sys.exit(1)

    #For batch processing
    extract_dir = sys.argv[1]
    out_path = os.path.join(sys.argv[2], f"{extract_dir}.json")

    print("Starting dataset processing...")
    process_dataset(extract_dir, out_path)

if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
import iresnet
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# Samples
NON_MATED_COMPARISONS = 50
MATED_COMPARISONS_PER_IDENTITY = 5

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
FOLDER_PATH = "Models"
MODEL_NAME = "ArcFace_R100_MS1MV3.pth"
weights = os.path.join(FOLDER_PATH, MODEL_NAME)

# Dataset configuration
DATASET_PATH = "Dataset2"

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
        # Try different image loading methods
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path).convert('RGB')
        else:
            # Use OpenCV for other formats
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_features(image_tensor):
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

def get_dataset_structure(dataset_path):
    """Get all identities and their corresponding images"""
    identities = {}
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return identities
    
    for identity_folder in os.listdir(dataset_path):
        identity_path = os.path.join(dataset_path, identity_folder)
        
        if os.path.isdir(identity_path):
            image_files = []
            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(file_path)
            
            if image_files:
                identities[identity_folder] = image_files
                print(f"Found identity '{identity_folder}' with {len(image_files)} images")
    
    print(f"Total identities found: {len(identities)}")
    return identities

def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors"""
    if features1 is None or features2 is None:
        return None
    
    # Ensure features are numpy arrays
    if torch.is_tensor(features1):
        features1 = features1.cpu().numpy()
    if torch.is_tensor(features2):
        features2 = features2.cpu().numpy()
    
    # Compute cosine similarity
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return float(similarity)

def get_image_paths_only(identities):
    """Get only the image paths without extracting features"""
    valid_identities = {}
    
    print("Validating image paths...")
    for identity, image_paths in tqdm(identities.items(), desc="Validating images"):
        valid_paths = []
        
        for image_path in image_paths:
            # Quick validation - just check if file exists and has valid extension
            if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_paths.append(image_path)
        
        if len(valid_paths) >= 2:  # Need at least 2 images for mated comparisons
            valid_identities[identity] = valid_paths
            print(f"Identity '{identity}': {len(valid_paths)} valid images")
    
    return valid_identities

def compute_mated_scores(valid_identities, max_pairs_per_identity=10):
    """Compute mated (same identity) similarity scores with on-demand feature extraction"""
    mated_scores = []
    
    print("Computing mated similarity scores...")
    for identity, image_paths in tqdm(valid_identities.items(), desc="Mated scores"):
        n_images = len(image_paths)
        
        if n_images < 2:
            continue
        
        # Get all possible pairs for this identity
        all_pairs = []
        for i in range(n_images):
            for j in range(i + 1, n_images):
                all_pairs.append((i, j))
        
        # Randomly sample pairs if there are too many
        if len(all_pairs) > max_pairs_per_identity:
            selected_pairs = random.sample(all_pairs, max_pairs_per_identity)
        else:
            selected_pairs = all_pairs
        
        # Extract features only for selected pairs and compute similarity
        for i, j in selected_pairs:
            # Load and extract features on-demand
            image1_tensor = load_image(image_paths[i])
            image2_tensor = load_image(image_paths[j])
            
            if image1_tensor is not None and image2_tensor is not None:
                features1 = extract_features(image1_tensor)
                features2 = extract_features(image2_tensor)
                
                if features1 is not None and features2 is not None:
                    similarity = compute_cosine_similarity(features1, features2)
                    if similarity is not None:
                        mated_scores.append(similarity)
                else:
                    print(f"Failed to extract features for pair: {image_paths[i]}, {image_paths[j]}")
            else:
                print(f"Failed to load images: {image_paths[i]}, {image_paths[j]}")
        
        print(f"Identity '{identity}': {len(selected_pairs)} comparisons from {len(all_pairs)} possible pairs")
    
    print(f"Total mated comparisons: {len(mated_scores)}")
    return mated_scores

def compute_non_mated_scores(valid_identities, max_comparisons=10000):
    """Compute non-mated (different identity) similarity scores with on-demand feature extraction"""
    non_mated_scores = []
    identities = list(valid_identities.keys())
    
    print("Computing non-mated similarity scores...")
    
    # Randomly sample pairs to avoid too many comparisons
    comparisons_made = 0
    max_attempts = max_comparisons * 10  # Prevent infinite loops
    attempts = 0
    
    while comparisons_made < max_comparisons and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two different identities
        id1, id2 = random.sample(identities, 2)
        
        # Randomly select one image from each identity
        img1_path = random.choice(valid_identities[id1])
        img2_path = random.choice(valid_identities[id2])
        
        # Load and extract features on-demand
        image1_tensor = load_image(img1_path)
        image2_tensor = load_image(img2_path)
        
        if image1_tensor is not None and image2_tensor is not None:
            features1 = extract_features(image1_tensor)
            features2 = extract_features(image2_tensor)
            
            if features1 is not None and features2 is not None:
                similarity = compute_cosine_similarity(features1, features2)
                if similarity is not None:
                    non_mated_scores.append(similarity)
                    comparisons_made += 1
        
        if comparisons_made % 1000 == 0 and comparisons_made > 0:
            print(f"Non-mated comparisons completed: {comparisons_made}")
    
    print(f"Total non-mated comparisons: {len(non_mated_scores)}")
    return non_mated_scores

def create_similarity_distribution_plot(mated_scores, non_mated_scores):
    """Create a beautiful distribution plot similar to the reference image"""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    '''
    # Create the distribution plots with filled areas
    # Non-mated scores (orange/red)
    ax.hist(non_mated_scores, bins=50, alpha=0.7, density=True, 
            color='#ff7f0e', label='Non-mated scores', edgecolor='white', linewidth=0.5)
    
    # Mated scores (blue)
    ax.hist(mated_scores, bins=50, alpha=0.7, density=True, 
            color='#1f77b4', label='Mated scores', edgecolor='white', linewidth=0.5)
    '''
    
    # Create x-axis for smooth curves
    x_min = min(min(mated_scores), min(non_mated_scores)) - 0.1
    x_max = max(max(mated_scores), max(non_mated_scores)) + 0.1
    x = np.linspace(x_min, x_max, 200)
    
    # Kernel density estimation
    try:
        non_mated_kde = stats.gaussian_kde(non_mated_scores)
        mated_kde = stats.gaussian_kde(mated_scores)
        
        # Plot smooth curves on top
        ax.plot(x, non_mated_kde(x), color='#d62728', linewidth=2, alpha=0.8)
        ax.plot(x, mated_kde(x), color='#1f77b4', linewidth=2, alpha=0.8)
    except:
        print("Could not create KDE curves")
    
    # Customize the plot
    ax.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Mated vs non-mated similarity scores', fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    
    # Customize legend
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper left')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Spines styling
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def analyze_and_plot_results(mated_scores, non_mated_scores):
    """Analyze and visualize the similarity scores"""
    print("\n=== SIMILARITY ANALYSIS RESULTS ===")
    
    # Basic statistics
    print(f"\nMated Scores Statistics:")
    print(f"  Count: {len(mated_scores)}")
    print(f"  Mean: {np.mean(mated_scores):.4f}")
    print(f"  Std: {np.std(mated_scores):.4f}")
    print(f"  Min: {np.min(mated_scores):.4f}")
    print(f"  Max: {np.max(mated_scores):.4f}")
    
    print(f"\nNon-Mated Scores Statistics:")
    print(f"  Count: {len(non_mated_scores)}")
    print(f"  Mean: {np.mean(non_mated_scores):.4f}")
    print(f"  Std: {np.std(non_mated_scores):.4f}")
    print(f"  Min: {np.min(non_mated_scores):.4f}")
    print(f"  Max: {np.max(non_mated_scores):.4f}")
    
    # Create the main similarity distribution plot
    create_similarity_distribution_plot(mated_scores, non_mated_scores)
    
    # Create additional analysis plots
    plt.figure(figsize=(10, 5))
    
    # Box plot
    plt.subplot(1, 2, 1)
    data_to_plot = [mated_scores, non_mated_scores]
    box_plot = plt.boxplot(data_to_plot, labels=['Mated', 'Non-Mated'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('#1f77b4')
    box_plot['boxes'][1].set_facecolor('#ff7f0e')
    plt.ylabel('Cosine Similarity Score')
    plt.title('Box Plot of Similarity Scores')
    plt.grid(True, alpha=0.3)
    
    # ROC-like analysis
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(min(min(mated_scores), min(non_mated_scores)), 
                           max(max(mated_scores), max(non_mated_scores)), 100)
    
    far_rates = []  # False Accept Rate
    frr_rates = []  # False Reject Rate
    
    for threshold in thresholds:
        # False accepts: non-mated scores above threshold
        far = sum(1 for score in non_mated_scores if score >= threshold) / len(non_mated_scores)
        # False rejects: mated scores below threshold
        frr = sum(1 for score in mated_scores if score < threshold) / len(mated_scores)
        
        far_rates.append(far)
        frr_rates.append(frr)
    
    plt.plot(thresholds, far_rates, label='FAR (False Accept Rate)', color='red', linewidth=2)
    plt.plot(thresholds, frr_rates, label='FRR (False Reject Rate)', color='blue', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR and FRR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find EER (Equal Error Rate)
    eer_idx = np.argmin(np.abs(np.array(far_rates) - np.array(frr_rates)))
    eer_threshold = thresholds[eer_idx]
    eer_rate = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
    
    print(f"\nEqual Error Rate (EER): {eer_rate:.4f} at threshold {eer_threshold:.4f}")
    
    return {
        'mated_stats': {
            'mean': np.mean(mated_scores),
            'std': np.std(mated_scores),
            'min': np.min(mated_scores),
            'max': np.max(mated_scores),
            'count': len(mated_scores)
        },
        'non_mated_stats': {
            'mean': np.mean(non_mated_scores),
            'std': np.std(non_mated_scores),
            'min': np.min(non_mated_scores),
            'max': np.max(non_mated_scores),
            'count': len(non_mated_scores)
        },
        'eer': eer_rate,
        'eer_threshold': eer_threshold
    }

def save_results(mated_scores, non_mated_scores, results_summary):
    """Save results to files"""
    # Save scores to text files
    with open('mated_scores.txt', 'w') as f:
        for score in mated_scores:
            f.write(f"{score}\n")
    
    with open('non_mated_scores.txt', 'w') as f:
        for score in non_mated_scores:
            f.write(f"{score}\n")
    
    # Save summary
    with open('similarity_analysis_summary.txt', 'w') as f:
        f.write("=== FACE RECOGNITION SIMILARITY ANALYSIS ===\n\n")
        f.write(f"Mated Scores (Same Identity):\n")
        f.write(f"  Count: {results_summary['mated_stats']['count']}\n")
        f.write(f"  Mean: {results_summary['mated_stats']['mean']:.4f}\n")
        f.write(f"  Std: {results_summary['mated_stats']['std']:.4f}\n")
        f.write(f"  Range: [{results_summary['mated_stats']['min']:.4f}, {results_summary['mated_stats']['max']:.4f}]\n\n")
        
        f.write(f"Non-Mated Scores (Different Identity):\n")
        f.write(f"  Count: {results_summary['non_mated_stats']['count']}\n")
        f.write(f"  Mean: {results_summary['non_mated_stats']['mean']:.4f}\n")
        f.write(f"  Std: {results_summary['non_mated_stats']['std']:.4f}\n")
        f.write(f"  Range: [{results_summary['non_mated_stats']['min']:.4f}, {results_summary['non_mated_stats']['max']:.4f}]\n\n")
        
        f.write(f"Equal Error Rate (EER): {results_summary['eer']:.4f}\n")
        f.write(f"EER Threshold: {results_summary['eer_threshold']:.4f}\n")
    
    print("Results saved to:")
    print("- mated_scores.txt")
    print("- non_mated_scores.txt") 
    print("- similarity_analysis_summary.txt")
    print("- similarity_distribution.png")
    print("- additional_analysis.png")

def main():
    """Main execution function"""
    print("Starting Face Recognition Similarity Analysis...")
    
    # Step 1: Get dataset structure
    identities = get_dataset_structure(DATASET_PATH)
    
    if not identities:
        print("No identities found in the dataset!")
        return
    
    # Step 2: Validate image paths (no feature extraction yet)
    valid_identities = get_image_paths_only(identities)
    print(f"Identities with 2+ images for mated comparisons: {len(valid_identities)}")
    
    if len(valid_identities) < 2:
        print("Need at least 2 identities with 2+ images each!")
        return
    
    # Step 3: Compute mated similarity scores (features extracted on-demand)
    mated_scores = compute_mated_scores(valid_identities, max_pairs_per_identity=MATED_COMPARISONS_PER_IDENTITY)
    
    # Step 4: Compute non-mated similarity scores (features extracted on-demand)
    non_mated_scores = compute_non_mated_scores(valid_identities, max_comparisons=NON_MATED_COMPARISONS)
    
    if not mated_scores or not non_mated_scores:
        print("Failed to compute similarity scores!")
        return
    
    # Step 5: Analyze and visualize results
    results_summary = analyze_and_plot_results(mated_scores, non_mated_scores)
    
    # Step 6: Save results
    save_results(mated_scores, non_mated_scores, results_summary)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
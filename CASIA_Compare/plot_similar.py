import argparse
import csv
import os
import shutil
import tempfile
import zipfile
import tarfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def extract_archive(archive_path, extract_to):
    """Extract zip or tar.gz archive to specified directory."""
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def find_image_at_path(base_dir, relative_path):
    """Find image file at the given relative path, searching at maximum depth."""
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Try the exact path first
    full_path = os.path.join(base_dir, relative_path)
    if os.path.exists(full_path):
        return full_path
    
    # Find all directories at maximum depth
    max_depth = 0
    deepest_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        depth = root[len(base_dir):].count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_dirs = [root]
        elif depth == max_depth:
            deepest_dirs.append(root)
    
    # Search for the image in deepest directories
    target_filename = os.path.basename(relative_path)
    
    for dir_path in deepest_dirs:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower() == target_filename.lower():
                    file_path = os.path.join(root, file)
                    if any(file_path.lower().endswith(ext) for ext in image_extensions):
                        return file_path
    
    # If not found by exact name, try to find any image matching the pattern
    for dir_path in deepest_dirs:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if target_filename.lower() in file.lower() or file.lower() in target_filename.lower():
                        return os.path.join(root, file)
    
    return None

def load_and_resize_image(image_path, max_size=(300, 300)):
    """Load image and resize it while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array to ensure proper dtype
            import numpy as np
            img_array = np.array(img)
            
            # Ensure the array is in the correct format for matplotlib
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_folder_and_filename(file_path):
    """Extract folder name and filename from path."""
    path_parts = file_path.replace('\\', '/').split('/')
    if len(path_parts) >= 2:
        return f"{path_parts[-2]}/{path_parts[-1]}"
    else:
        return path_parts[-1]

def create_multi_comparison_figure(samples_data, output_path):
    """Create a single figure with multiple image comparisons in a grid layout."""
    n_samples = len(samples_data)
    
    # Create figure with grid layout: 2 rows per sample (casia, other), n_samples columns
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (casia_img, other_img, similarity_score, query_path, casia_path) in enumerate(samples_data):
        # CASIA image (top row)
        if casia_img is not None:
            axes[0, col].imshow(casia_img)
        else:
            axes[0, col].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', 
                            transform=axes[0, col].transAxes, fontsize=10, color='red')
        
        # Extract folder name as identity and filename
        folder_filename = get_folder_and_filename(casia_path)
        parts = folder_filename.split('/')
        if len(parts) == 2:
            identity, filename = parts
        else:
            identity = "Unknown"
            filename = parts[0] if parts else "Unknown"
        
        axes[0, col].set_title(f'Identity: {identity}\nImage: {filename}', fontsize=9, fontweight='bold')
        axes[0, col].axis('off')
        
        # Query/Other image (bottom row)
        if other_img is not None:
            axes[1, col].imshow(other_img)
        else:
            axes[1, col].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', 
                            transform=axes[1, col].transAxes, fontsize=10, color='red')
        
        # Extract folder name as identity and filename for query
        folder_filename_query = get_folder_and_filename(query_path)
        parts_query = folder_filename_query.split('/')
        if len(parts_query) == 2:
            identity_query, filename_query = parts_query
        else:
            identity_query = "Unknown"
            filename_query = parts_query[0] if parts_query else "Unknown"
            
        axes[1, col].set_title(f'Identity: {identity_query}\nImage: {filename_query}', fontsize=9, fontweight='bold')
        axes[1, col].axis('off')
        
        # Add similarity score in the center between the two images
        fig.text((col + 0.5) / n_samples, 0.5, f'{similarity_score:.4f}', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate image similarity comparison figures')
    parser.add_argument('csv_file', help='Path to CSV file with similarity data')
    parser.add_argument('--casia-file', default='CASIA.zip', 
                       help='Path to CASIA archive file (default: CASIA.zip)')
    parser.add_argument('--other-file', required=True,
                       help='Path to other archive file')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to process (default: 5)')
    parser.add_argument('--output-dir', default='comparison_figures',
                       help='Output directory for figures (default: comparison_figures)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temporary directories for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        casia_extract_dir = os.path.join(temp_dir, 'casia')
        other_extract_dir = os.path.join(temp_dir, 'other')
        
        os.makedirs(casia_extract_dir)
        os.makedirs(other_extract_dir)
        
        print(f"Extracting {args.casia_file}...")
        extract_archive(args.casia_file, casia_extract_dir)
        
        print(f"Extracting {args.other_file}...")
        extract_archive(args.other_file, other_extract_dir)
        
        # Read CSV file and collect all samples data
        samples_data = []
        with open(args.csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                if len(samples_data) >= args.samples:
                    break
                
                query_path = row['query_path'].strip('"')
                casia_path = row['casia_path'].strip('"')
                similarity_score = float(row['similarity_score'])
                
                print(f"Processing sample {len(samples_data) + 1}/{args.samples}")
                print(f"  Query: {query_path}")
                print(f"  CASIA: {casia_path}")
                print(f"  Similarity: {similarity_score:.4f}")
                
                # Find images
                casia_img_path = find_image_at_path(casia_extract_dir, casia_path)
                other_img_path = find_image_at_path(other_extract_dir, query_path)
                
                if casia_img_path:
                    print(f"  Found CASIA image: {casia_img_path}")
                else:
                    print(f"  CASIA image not found: {casia_path}")
                
                if other_img_path:
                    print(f"  Found other image: {other_img_path}")
                else:
                    print(f"  Other image not found: {query_path}")
                
                # Load images
                casia_img = load_and_resize_image(casia_img_path) if casia_img_path else None
                other_img = load_and_resize_image(other_img_path) if other_img_path else None
                
                # Store sample data
                samples_data.append((casia_img, other_img, similarity_score, query_path, casia_path))
                print()
        
        # Create single comparison figure with all samples
        if samples_data:
            # Extract base name from other file (without extension)
            other_file_base = os.path.splitext(os.path.basename(args.other_file))[0]
            # Remove .tar if it's a .tar.gz file
            if other_file_base.endswith('.tar'):
                other_file_base = other_file_base[:-4]
            
            output_filename = f"comparison_{other_file_base}_{len(samples_data)}_samples.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            create_multi_comparison_figure(samples_data, output_path)
            print(f"Created comparison grid: {output_path}")
    
    print(f"Processing complete! Generated 1 comparison figure with {len(samples_data)} samples in '{args.output_dir}'")

if __name__ == "__main__":
    main()
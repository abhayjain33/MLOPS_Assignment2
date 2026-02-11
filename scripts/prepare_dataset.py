"""
Prepare dataset from PetImages directory.
Cleans corrupted images and organizes data for training.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def is_valid_image(image_path):
    """Check if image file is valid and can be opened."""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def prepare_dataset(source_dir, target_dir):
    """
    Prepare dataset by copying valid images from source to target.
    
    Args:
        source_dir: Path to PetImages directory
        target_dir: Path to output directory (data/raw)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    (target_path / "cats").mkdir(parents=True, exist_ok=True)
    (target_path / "dogs").mkdir(parents=True, exist_ok=True)
    
    stats = {"cats": {"total": 0, "valid": 0, "invalid": 0},
             "dogs": {"total": 0, "valid": 0, "invalid": 0}}
    
    # Process each class
    for class_name in ["Cat", "Dog"]:
        source_class_dir = source_path / class_name
        target_class_name = class_name.lower() + "s"
        target_class_dir = target_path / target_class_name
        
        if not source_class_dir.exists():
            print(f"Warning: {source_class_dir} does not exist")
            continue
        
        print(f"\nProcessing {class_name} images...")
        image_files = list(source_class_dir.glob("*.jpg"))
        stats[target_class_name]["total"] = len(image_files)
        
        for img_file in tqdm(image_files, desc=f"Copying {class_name}s"):
            if is_valid_image(img_file):
                # Copy valid image with new name
                new_name = f"{target_class_name[:-1]}_{stats[target_class_name]['valid']:05d}.jpg"
                target_file = target_class_dir / new_name
                
                try:
                    shutil.copy2(img_file, target_file)
                    stats[target_class_name]["valid"] += 1
                except Exception as e:
                    print(f"Error copying {img_file}: {e}")
                    stats[target_class_name]["invalid"] += 1
            else:
                stats[target_class_name]["invalid"] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Preparation Summary")
    print("="*60)
    
    for class_name in ["cats", "dogs"]:
        print(f"\n{class_name.capitalize()}:")
        print(f"  Total images: {stats[class_name]['total']}")
        print(f"  Valid images: {stats[class_name]['valid']}")
        print(f"  Invalid/corrupted: {stats[class_name]['invalid']}")
    
    total_valid = stats["cats"]["valid"] + stats["dogs"]["valid"]
    print(f"\nTotal valid images: {total_valid}")
    print(f"Dataset saved to: {target_path}")
    print("="*60)


if __name__ == "__main__":
    # Default paths
    source = "Data/archive/PetImages"
    target = "data/raw"
    
    print("Starting dataset preparation...")
    print(f"Source: {source}")
    print(f"Target: {target}")
    
    if not Path(source).exists():
        print(f"\nError: Source directory not found: {source}")
        print("Please update the path in this script or provide it as an argument.")
        exit(1)
    
    prepare_dataset(source, target)
    print("\n✓ Dataset preparation completed!")

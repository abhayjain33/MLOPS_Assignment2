"""
Utility script to create dummy data for testing.
Use this if you don't have the Kaggle dataset yet.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def create_dummy_dataset(output_dir: str, num_images_per_class: int = 100):
    """
    Create a dummy cats vs dogs dataset for testing.
    
    Args:
        output_dir: Directory to save the dataset
        num_images_per_class: Number of images to create per class
    """
    output_path = Path(output_dir)
    
    # Create directories
    cats_dir = output_path / "cats"
    dogs_dir = output_path / "dogs"
    
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy dataset in {output_path}")
    print(f"Generating {num_images_per_class} images per class...")
    
    # Create cat images (more uniform patterns)
    for i in tqdm(range(num_images_per_class), desc="Creating cat images"):
        # Create a random image with more uniform colors (simulating cats)
        img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(cats_dir / f"cat_{i:04d}.jpg")
    
    # Create dog images (more varied patterns)
    for i in tqdm(range(num_images_per_class), desc="Creating dog images"):
        # Create a random image with more varied colors (simulating dogs)
        img_array = np.random.randint(50, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(dogs_dir / f"dog_{i:04d}.jpg")
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Cats: {len(list(cats_dir.glob('*.jpg')))} images")
    print(f"  Dogs: {len(list(dogs_dir.glob('*.jpg')))} images")
    print(f"  Total: {len(list(output_path.rglob('*.jpg')))} images")
    print(f"\nNote: This is dummy data for testing. For real training,")
    print(f"download the Cats and Dogs dataset from Kaggle.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dummy dataset for testing")
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for the dataset'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='Number of images per class'
    )
    
    args = parser.parse_args()
    create_dummy_dataset(args.output, args.num_images)

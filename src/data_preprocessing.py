"""
Data preprocessing module for Cats vs Dogs classification.
Handles data loading, preprocessing, augmentation, and splitting.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np


class CatsDogsDataPreprocessor:
    """Preprocessor for Cats vs Dogs dataset."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str, img_size: int = 224):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to raw dataset
            processed_data_path: Path to save processed data
            img_size: Target image size (default: 224x224)
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.img_size = img_size
        
    def get_transforms(self, augment: bool = False) -> transforms.Compose:
        """
        Get image transformation pipeline.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Composed transforms
        """
        if augment:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        transform = self.get_transforms(augment=False)
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    
    def split_dataset(self, 
                     dataset: datasets.ImageFolder,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Tuple:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: ImageFolder dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self,
                          batch_size: int = 32,
                          num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for train, val, and test sets.
        
        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load full dataset with augmentation for training
        full_dataset = datasets.ImageFolder(
            root=self.raw_data_path,
            transform=self.get_transforms(augment=True)
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(full_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self) -> list:
        """
        Get class names from dataset.
        
        Returns:
            List of class names
        """
        dataset = datasets.ImageFolder(root=self.raw_data_path)
        return dataset.classes


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized numpy array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


if __name__ == "__main__":
    # Example usage
    raw_path = "data/raw"
    processed_path = "data/processed"
    
    preprocessor = CatsDogsDataPreprocessor(raw_path, processed_path)
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {preprocessor.get_class_names()}")
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

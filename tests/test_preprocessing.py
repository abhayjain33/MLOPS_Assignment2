"""
Unit tests for data preprocessing functions.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from src.data_preprocessing import (
    CatsDogsDataPreprocessor,
    denormalize_image
)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create directory structure
    raw_path = Path(temp_dir) / "raw"
    processed_path = Path(temp_dir) / "processed"
    
    # Create class directories
    (raw_path / "cats").mkdir(parents=True)
    (raw_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    for i in range(5):
        # Create random RGB image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(raw_path / "cats" / f"cat_{i}.jpg")
        img.save(raw_path / "dogs" / f"dog_{i}.jpg")
    
    yield raw_path, processed_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_preprocessor_initialization(temp_data_dir):
    """Test preprocessor initialization."""
    raw_path, processed_path = temp_data_dir
    
    preprocessor = CatsDogsDataPreprocessor(
        str(raw_path),
        str(processed_path),
        img_size=224
    )
    
    assert preprocessor.img_size == 224
    assert preprocessor.raw_data_path == raw_path
    assert preprocessor.processed_data_path == processed_path


def test_get_transforms_no_augmentation(temp_data_dir):
    """Test transforms without augmentation."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    transform = preprocessor.get_transforms(augment=False)
    
    # Test transform
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    output = transform(img)
    
    # Check output shape
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_get_transforms_with_augmentation(temp_data_dir):
    """Test transforms with augmentation."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    transform = preprocessor.get_transforms(augment=True)
    
    # Test transform
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    output = transform(img)
    
    # Check output shape
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_preprocess_image(temp_data_dir):
    """Test single image preprocessing."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    # Get path to test image
    image_path = raw_path / "cats" / "cat_0.jpg"
    
    # Preprocess
    output = preprocessor.preprocess_image(str(image_path))
    
    # Verify output
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_split_dataset(temp_data_dir):
    """Test dataset splitting."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    from torchvision import datasets
    dataset = datasets.ImageFolder(root=raw_path)
    
    train_ds, val_ds, test_ds = preprocessor.split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Check sizes
    total_size = len(dataset)
    assert len(train_ds) == int(0.8 * total_size)
    assert len(val_ds) == int(0.1 * total_size)
    assert len(test_ds) == total_size - len(train_ds) - len(val_ds)


def test_split_dataset_invalid_ratios(temp_data_dir):
    """Test dataset splitting with invalid ratios."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    from torchvision import datasets
    dataset = datasets.ImageFolder(root=raw_path)
    
    # Should raise assertion error
    with pytest.raises(AssertionError):
        preprocessor.split_dataset(
            dataset,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.1  # Sum is not 1.0
        )


def test_get_class_names(temp_data_dir):
    """Test getting class names."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    class_names = preprocessor.get_class_names()
    
    assert len(class_names) == 2
    assert 'cats' in class_names
    assert 'dogs' in class_names


def test_create_dataloaders(temp_data_dir):
    """Test dataloader creation."""
    raw_path, processed_path = temp_data_dir
    preprocessor = CatsDogsDataPreprocessor(str(raw_path), str(processed_path))
    
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )
    
    # Check dataloaders exist
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # Test getting a batch
    images, labels = next(iter(train_loader))
    assert images.shape[0] <= 2  # Batch size
    assert images.shape[1:] == (3, 224, 224)
    assert labels.shape[0] <= 2


def test_denormalize_image():
    """Test image denormalization."""
    # Create normalized tensor
    tensor = torch.randn(3, 224, 224)
    
    # Denormalize
    denorm = denormalize_image(tensor)
    
    # Check output
    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (224, 224, 3)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0


def test_preprocessor_with_different_image_size(temp_data_dir):
    """Test preprocessor with custom image size."""
    raw_path, processed_path = temp_data_dir
    
    custom_size = 128
    preprocessor = CatsDogsDataPreprocessor(
        str(raw_path),
        str(processed_path),
        img_size=custom_size
    )
    
    image_path = raw_path / "cats" / "cat_0.jpg"
    output = preprocessor.preprocess_image(str(image_path))
    
    assert output.shape == (3, custom_size, custom_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

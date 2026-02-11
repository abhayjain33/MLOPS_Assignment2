"""
Unit tests for model inference.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil
from io import BytesIO

from src.model import CatsDogsClassifier, create_model
from src.inference import ModelInference, get_model_info


@pytest.fixture
def dummy_model_path():
    """Create a dummy model file for testing."""
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "test_model.pth"
    
    # Create and save a model
    model = create_model(num_classes=2)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 10,
        'best_val_acc': 95.5
    }, model_path)
    
    yield str(model_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_image():
    """Create a dummy image for testing."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    temp_dir = tempfile.mkdtemp()
    img_path = Path(temp_dir) / "test_image.jpg"
    img.save(img_path)
    
    yield str(img_path), img
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_model_creation():
    """Test model creation."""
    model = create_model(num_classes=2, dropout_rate=0.5)
    
    assert isinstance(model, CatsDogsClassifier)
    assert model.fc3.out_features == 2


def test_model_forward_pass():
    """Test model forward pass."""
    model = create_model()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (1, 2)


def test_model_predict():
    """Test model prediction method."""
    model = create_model()
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Predict
    probs = model.predict(x)
    
    # Check output
    assert probs.shape == (1, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_inference_initialization(dummy_model_path):
    """Test ModelInference initialization."""
    # Create dummy data directory
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "raw"
    (data_path / "cats").mkdir(parents=True)
    (data_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(data_path / "cats" / f"cat_{i}.jpg")
        img.save(data_path / "dogs" / f"dog_{i}.jpg")
    
    try:
        # Initialize inference
        inference = ModelInference(dummy_model_path)
        
        assert inference.model is not None
        assert inference.device in ['cpu', 'cuda']
        assert len(inference.class_names) == 2
    finally:
        shutil.rmtree(temp_dir)


def test_load_model(dummy_model_path):
    """Test model loading."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "raw"
    (data_path / "cats").mkdir(parents=True)
    (data_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(data_path / "cats" / "cat_0.jpg")
    
    try:
        inference = ModelInference(dummy_model_path)
        model = inference.load_model(dummy_model_path)
        
        assert isinstance(model, CatsDogsClassifier)
        # Model is in eval mode in __init__, not in load_model
    finally:
        shutil.rmtree(temp_dir)


def test_predict_image(dummy_model_path, dummy_image):
    """Test single image prediction."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "raw"
    (data_path / "cats").mkdir(parents=True)
    (data_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(data_path / "cats" / "cat_0.jpg")
    
    try:
        image_path, _ = dummy_image
        inference = ModelInference(dummy_model_path)
        
        result = inference.predict_image(image_path)
        
        # Check result structure
        assert 'cats' in result
        assert 'dogs' in result
        assert 'predicted_class' in result
        assert 'confidence' in result
        
        # Check values
        assert result['predicted_class'] in ['cats', 'dogs']
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['cats'] <= 1
        assert 0 <= result['dogs'] <= 1
        assert abs(result['cats'] + result['dogs'] - 1.0) < 1e-5
    finally:
        shutil.rmtree(temp_dir)


def test_predict_from_bytes(dummy_model_path, dummy_image):
    """Test prediction from image bytes."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "raw"
    (data_path / "cats").mkdir(parents=True)
    (data_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(data_path / "cats" / "cat_0.jpg")
    
    try:
        _, img = dummy_image
        
        # Convert image to bytes
        buf = BytesIO()
        img.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        
        inference = ModelInference(dummy_model_path)
        result = inference.predict_from_bytes(image_bytes)
        
        # Check result structure
        assert 'cats' in result
        assert 'dogs' in result
        assert 'predicted_class' in result
        assert 'confidence' in result
        
        # Check values
        assert result['predicted_class'] in ['cats', 'dogs']
        assert 0 <= result['confidence'] <= 1
    finally:
        shutil.rmtree(temp_dir)


def test_predict_batch(dummy_model_path, dummy_image):
    """Test batch prediction."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "raw"
    (data_path / "cats").mkdir(parents=True)
    (data_path / "dogs").mkdir(parents=True)
    
    # Create dummy images
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(data_path / "cats" / "cat_0.jpg")
    
    try:
        image_path, _ = dummy_image
        inference = ModelInference(dummy_model_path)
        
        # Predict batch
        image_paths = [image_path, image_path]
        results = inference.predict_batch(image_paths)
        
        # Check results
        assert len(results) == 2
        for result in results:
            assert 'predicted_class' in result
            assert 'confidence' in result
    finally:
        shutil.rmtree(temp_dir)


def test_get_model_info(dummy_model_path):
    """Test getting model information."""
    info = get_model_info(dummy_model_path)
    
    assert 'model_path' in info
    assert 'file_size_mb' in info
    assert 'epoch' in info
    assert 'best_val_accuracy' in info
    
    assert info['epoch'] == 10
    assert info['best_val_accuracy'] == 95.5


def test_model_parameter_count():
    """Test model has expected number of parameters."""
    model = create_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check that model has parameters
    assert total_params > 0
    assert trainable_params > 0
    assert total_params == trainable_params  # All params should be trainable


def test_model_output_range():
    """Test that model outputs are in valid range after softmax."""
    model = create_model()
    model.eval()
    
    x = torch.randn(4, 3, 224, 224)
    probs = model.predict(x)
    
    # Check probabilities sum to 1
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)
    
    # Check all probabilities are between 0 and 1
    assert (probs >= 0).all()
    assert (probs <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

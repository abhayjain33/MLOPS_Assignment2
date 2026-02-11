"""
Model inference utilities for predictions.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

from src.model import CatsDogsClassifier
from src.data_preprocessing import CatsDogsDataPreprocessor


class ModelInference:
    """Handle model inference for single images or batches."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to saved model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        self.class_names = ['cats', 'dogs']
        
        # Initialize preprocessor
        self.preprocessor = CatsDogsDataPreprocessor("data/raw", "data/processed")
        
    def load_model(self, model_path: str) -> CatsDogsClassifier:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        model = CatsDogsClassifier(num_classes=2)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def predict_image(self, image_path: str) -> Dict[str, float]:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with class probabilities
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Convert to dictionary
        probs = probabilities.cpu().numpy()[0]
        result = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        # Add predicted class
        predicted_idx = np.argmax(probs)
        result['predicted_class'] = self.class_names[predicted_idx]
        result['confidence'] = float(probs[predicted_idx])
        
        return result
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """
        Predict class from image bytes (for API use).
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with class probabilities
        """
        from io import BytesIO
        
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        transform = self.preprocessor.get_transforms(augment=False)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Convert to dictionary
        probs = probabilities.cpu().numpy()[0]
        result = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        # Add predicted class
        predicted_idx = np.argmax(probs)
        result['predicted_class'] = self.class_names[predicted_idx]
        result['confidence'] = float(probs[predicted_idx])
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries with predictions
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        return results


def get_model_info(model_path: str) -> Dict:
    """
    Get information about saved model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model information
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    info = {
        'model_path': model_path,
        'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
    }
    
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'best_val_acc' in checkpoint:
            info['best_val_accuracy'] = checkpoint['best_val_acc']
        if 'optimizer_state_dict' in checkpoint:
            info['has_optimizer_state'] = True
    
    return info


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        model_path = "models/best_model.pth"
        image_path = sys.argv[1]
        
        print(f"Loading model from {model_path}")
        inference = ModelInference(model_path)
        
        print(f"Predicting image: {image_path}")
        result = inference.predict_image(image_path)
        
        print("\nPrediction results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python src/inference.py <image_path>")

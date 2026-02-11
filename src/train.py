"""
Training script with MLflow experiment tracking.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.model import create_model, CatsDogsClassifier
from src.data_preprocessing import CatsDogsDataPreprocessor


class Trainer:
    """Model trainer with MLflow tracking."""
    
    def __init__(self, 
                 model: CatsDogsClassifier,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = None,
                 learning_rate: float = 0.001,
                 num_epochs: int = 20):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(self.train_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int = None) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        desc = f"Epoch {epoch+1}/{self.num_epochs} [Val]" if epoch is not None else "Validation"
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=desc)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(self.val_loader):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self) -> tuple:
        """Test the model and generate confusion matrix."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate accuracy
        correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        test_acc = 100. * correct / len(all_labels)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return test_acc, cm, all_preds, all_labels
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['cats', 'dogs'],
                   yticklabels=['cats', 'dogs'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def train(self):
        """Full training loop with MLflow tracking."""
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model("models/best_model.pth", epoch, val_acc)
                print(f"✓ Best model saved with val_acc: {val_acc:.2f}%")
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_model(self, path: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': val_acc,
        }, path)


def main(args):
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    
    # Initialize MLflow
    mlflow.set_experiment("cats-dogs-classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'dropout_rate': args.dropout,
            'img_size': 224,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        })
        
        # Prepare data
        print("Preparing data loaders...")
        preprocessor = CatsDogsDataPreprocessor(args.data_path, "data/processed")
        train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create model
        print("Creating model...")
        model = create_model(num_classes=2, dropout_rate=args.dropout)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        
        # Train
        trainer.train()
        
        # Test
        print("\nEvaluating on test set...")
        test_acc, cm, preds, labels = trainer.test()
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Log test accuracy
        mlflow.log_metric('test_acc', test_acc)
        
        # Plot and log artifacts
        print("Generating plots...")
        curves_path = trainer.plot_training_curves("artifacts/training_curves.png")
        cm_path = trainer.plot_confusion_matrix(cm, "artifacts/confusion_matrix.png")
        
        mlflow.log_artifact(curves_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact("models/best_model.pth")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print("\n✓ Training completed successfully!")
        print(f"✓ Best validation accuracy: {trainer.best_val_acc:.2f}%")
        print(f"✓ Test accuracy: {test_acc:.2f}%")
        print(f"✓ Model saved to: models/best_model.pth")
        print(f"✓ MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument('--data_path', type=str, default='data/raw',
                       help='Path to raw dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)

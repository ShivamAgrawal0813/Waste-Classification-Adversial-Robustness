"""
Utility functions for the waste classification project.
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from PIL import Image

from src.config import SEED, OUTPUTS_DIR, METRICS_FILE, CLASS_NAMES, IDX_TO_CLASS


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_metrics(metrics: Dict, filepath: str = METRICS_FILE):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str = METRICS_FILE) -> Dict:
    """Load metrics from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def calculate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Calculate accuracy of model on dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         save_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_accuracy_vs_epsilon(epsilons: List[float], clean_acc: float,
                            fgsm_accs: List[float], pgd_accs: List[float],
                            save_path: str):
    """Plot accuracy vs epsilon for FGSM and PGD attacks."""
    plt.figure(figsize=(10, 6))
    plt.plot([0], [clean_acc], 'o-', label='Clean', linewidth=2, markersize=8)
    plt.plot(epsilons, fgsm_accs, 's-', label='FGSM', linewidth=2, markersize=6)
    plt.plot(epsilons, pgd_accs, '^-', label='PGD', linewidth=2, markersize=6)
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy vs Adversarial Perturbation Strength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy vs epsilon plot saved to {save_path}")


def save_model(model: nn.Module, filepath: str, epoch: int = None, 
              optimizer: torch.optim.Optimizer = None, 
              metrics: Dict = None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, device: torch.device,
              optimizer: torch.optim.Optimizer = None) -> Tuple[nn.Module, int, Optional[Dict]]:
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', None)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    return model, epoch, metrics


def get_predictions(model: nn.Module, dataloader: DataLoader, 
                   device: torch.device) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Get predictions and true labels from model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_labels, all_preds, all_probs


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Denormalize a tensor image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def save_sample_predictions(model: nn.Module, dataloader: DataLoader, 
                           device: torch.device, num_samples: int = 10,
                           save_dir: str = None):
    """Save sample predictions with images."""
    if save_dir is None:
        save_dir = os.path.join(OUTPUTS_DIR, "sample_predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if samples_saved >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if samples_saved >= num_samples:
                    break
                
                img = images[i].cpu()
                label = labels[i].item()
                pred = predicted[i].item()
                prob = probs[i][pred].item()
                
                # Denormalize image
                from src.config import MEAN, STD
                img_denorm = denormalize(img, MEAN, STD)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                img_np = img_denorm.permute(1, 2, 0).numpy()
                
                # Save image
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np)
                plt.title(f"True: {IDX_TO_CLASS[label]}\nPred: {IDX_TO_CLASS[pred]} ({prob:.2%})")
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f"sample_{samples_saved}.png"))
                plt.close()
                
                samples_saved += 1


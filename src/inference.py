"""
Single image inference helper for waste classification.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Dict

from src.config import (
    MODEL_NAME, NUM_CLASSES, PRETRAINED, IMAGE_SIZE, RESIZE_SIZE, MEAN, STD,
    CLASS_NAMES, IDX_TO_CLASS, DEVICE
)
from src.train import get_model
from src.utils import get_device, denormalize
from src.attacks import fgsm_attack, pgd_attack
import numpy as np


def load_image(image_path: str) -> Image.Image:
    """Load image from file path."""
    image = Image.open(image_path).convert('RGB')
    return image


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image
        image_size: Target image size
    
    Returns:
        Preprocessed tensor
    """
    # Maintain aspect ratio for resizing
    resize_size = int(image_size * (RESIZE_SIZE / IMAGE_SIZE))
    
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model: nn.Module, image_tensor: torch.Tensor, 
           device: torch.device, top_k: int = 3) -> Dict:
    """
    Get prediction from model.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    
    predictions = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        predictions.append({
            'class': IDX_TO_CLASS[idx],
            'class_idx': idx,
            'confidence': prob
        })
    
    return {
        'predictions': predictions,
        'predicted_class': predictions[0]['class'],
        'confidence': predictions[0]['confidence']
    }


def generate_adversarial_examples(model: nn.Module, image_tensor: torch.Tensor,
                                  label: int, device: torch.device) -> Dict:
    """
    Generate FGSM and PGD adversarial examples.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        label: True label (or predicted label if true label unknown)
        device: Device to run on
    
    Returns:
        Dictionary with adversarial examples and predictions
    """
    from src.config import EPS_FGSM, EPS_PGD, PGD_ALPHA, PGD_STEPS_EVAL
    
    image_tensor = image_tensor.to(device)
    label_tensor = torch.tensor([label]).to(device)
    
    # Get clean prediction first
    clean_pred = predict(model, image_tensor, device, top_k=1)
    predicted_label = clean_pred['predictions'][0]['class_idx']
    
    # Use predicted label for attacks (since we might not have true label)
    label_tensor = torch.tensor([predicted_label]).to(device)
    
    # Generate FGSM attack
    model.eval()
    fgsm_image = fgsm_attack(model, image_tensor, label_tensor, epsilon=EPS_FGSM)
    fgsm_pred = predict(model, fgsm_image, device, top_k=1)
    
    # Generate PGD attack
    pgd_image = pgd_attack(
        model, image_tensor, label_tensor,
        epsilon=EPS_PGD,
        alpha=PGD_ALPHA,
        num_steps=PGD_STEPS_EVAL
    )
    pgd_pred = predict(model, pgd_image, device, top_k=1)
    
    return {
        'fgsm_image': fgsm_image,
        'fgsm_prediction': fgsm_pred,
        'pgd_image': pgd_image,
        'pgd_prediction': pgd_pred,
    }


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image array.
    
    Args:
        tensor: Image tensor (C, H, W) or (1, C, H, W)
    
    Returns:
        Numpy array (H, W, C) in range [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    img = denormalize(tensor, MEAN, STD)
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    return img_np


def visualize_perturbation(original: torch.Tensor, adversarial: torch.Tensor,
                          amplification: float = 10.0) -> np.ndarray:
    """
    Visualize adversarial perturbation.
    
    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor
        amplification: Amplification factor for visualization
    
    Returns:
        Amplified perturbation as numpy array
    """
    if original.dim() == 4:
        original = original.squeeze(0)
    if adversarial.dim() == 4:
        adversarial = adversarial.squeeze(0)
    
    # Calculate perturbation
    perturbation = adversarial - original
    
    # Amplify and normalize
    perturbation = perturbation * amplification
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
    
    # Convert to image
    perturbation_np = perturbation.permute(1, 2, 0).cpu().numpy()
    
    return perturbation_np


def load_model_for_inference(model_path: str, model_name: str = MODEL_NAME, 
                            device: torch.device = None) -> nn.Module:
    """
    Load model for inference.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name of the architecture
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = get_device()
    
    model = get_model(model_name, NUM_CLASSES, PRETRAINED)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


"""
Adversarial attack implementations: FGSM and PGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.config import EPS_FGSM, EPS_PGD, PGD_ALPHA, PGD_STEPS_EVAL, RANDOM_START


def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
               epsilon: float = EPS_FGSM) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: PyTorch model
        images: Input images (batch, channels, height, width)
        labels: True labels
        epsilon: Attack strength (default from config)
    
    Returns:
        Adversarial examples
    """
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradient
    data_grad = images.grad.data
    
    # Create adversarial examples
    perturbed_images = images + epsilon * data_grad.sign()
    
    # Clip to [0, 1] range (assuming images are normalized)
    # For normalized images, we need to clip in normalized space
    # But typically we work in [0,1] space then normalize
    # Here we assume images are already normalized, so we clip perturbations
    perturbed_images = torch.clamp(perturbed_images, -2.0, 2.0)  # Wide range for normalized
    
    return perturbed_images.detach()


def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
              epsilon: float = EPS_PGD, alpha: float = PGD_ALPHA,
              num_steps: int = PGD_STEPS_EVAL,
              random_start: bool = RANDOM_START) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: PyTorch model
        images: Input images (batch, channels, height, width)
        labels: True labels
        epsilon: Attack strength (L_inf norm bound)
        alpha: Step size
        num_steps: Number of PGD steps
        random_start: Whether to start from random perturbation
    
    Returns:
        Adversarial examples
    """
    images = images.clone().detach()
    
    # Initialize perturbation
    if random_start:
        # Start from random point within epsilon ball
        perturbed_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, -2.0, 2.0)
    else:
        perturbed_images = images.clone()
    
    perturbed_images = perturbed_images.requires_grad_(True)
    
    for _ in range(num_steps):
        # Forward pass
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        if perturbed_images.grad is not None:
            perturbed_images.grad.zero_()
        loss.backward()
        
        # Get gradient
        data_grad = perturbed_images.grad.data
        
        # Update perturbation
        perturbed_images = perturbed_images + alpha * data_grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = images + eta
        
        # Clip to valid range
        perturbed_images = torch.clamp(perturbed_images, -2.0, 2.0)
        
        # Detach and re-attach gradient
        perturbed_images = perturbed_images.detach().requires_grad_(True)
    
    return perturbed_images.detach()


def pgd_attack_training(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                       epsilon: float = EPS_PGD, alpha: float = PGD_ALPHA,
                       num_steps: int = 7, random_start: bool = RANDOM_START) -> torch.Tensor:
    """
    PGD attack for adversarial training (fewer steps for efficiency).
    
    Args:
        model: PyTorch model
        images: Input images
        labels: True labels
        epsilon: Attack strength
        alpha: Step size
        num_steps: Number of PGD steps (typically 7 for training)
        random_start: Whether to start from random perturbation
    
    Returns:
        Adversarial examples
    """
    return pgd_attack(model, images, labels, epsilon, alpha, num_steps, random_start)


def evaluate_attack(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   device: torch.device, attack_type: str = 'fgsm',
                   epsilon: float = EPS_FGSM, alpha: float = PGD_ALPHA,
                   num_steps: int = PGD_STEPS_EVAL) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate model robustness against adversarial attacks.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        attack_type: 'fgsm' or 'pgd'
        epsilon: Attack strength
        alpha: Step size (for PGD)
        num_steps: Number of steps (for PGD)
    
    Returns:
        Tuple of (accuracy, all_predictions, all_labels)
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        if attack_type.lower() == 'fgsm':
            adv_images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type.lower() == 'pgd':
            adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    return accuracy, torch.tensor(all_preds), torch.tensor(all_labels)


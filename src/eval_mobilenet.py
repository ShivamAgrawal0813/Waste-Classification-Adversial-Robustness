"""
Evaluation script for MobileNet-V2.
Evaluates clean accuracy, FGSM, and PGD attacks, and saves plots.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    NUM_CLASSES, MOBILENET_CLEAN_PATH, MOBILENET_METRICS_FILE,
    EPS_FGSM, EPS_PGD, PGD_ALPHA, PGD_STEPS_EVAL,
    ACCURACY_VS_EPS_EPSILONS, OUTPUTS_DIR,
    MOBILENET_CONFUSION_CLEAN, MOBILENET_CONFUSION_PGD, MOBILENET_ACC_VS_EPS
)
from src.dataset import get_dataloaders
from src.utils import (
    get_device, calculate_accuracy, load_model, get_predictions,
    plot_confusion_matrix, plot_accuracy_vs_epsilon
)
from src.train import get_model
from src.attacks import evaluate_attack


def evaluate_accuracy_vs_epsilon(model: nn.Module, test_loader, device: torch.device,
                                 epsilons: list, model_name: str = "MobileNetV2") -> dict:
    """Evaluate model accuracy for different epsilon values."""
    print(f"\nEvaluating accuracy vs epsilon for {model_name}...")
    
    clean_acc = calculate_accuracy(model, test_loader, device)
    fgsm_accs = []
    pgd_accs = []
    
    for eps in epsilons:
        if eps == 0:
            fgsm_accs.append(clean_acc)
            pgd_accs.append(clean_acc)
        else:
            print(f"  Epsilon: {eps:.4f}...", end=' ')
            fgsm_acc, _, _ = evaluate_attack(
                model, test_loader, device, attack_type='fgsm', epsilon=eps
            )
            pgd_acc, _, _ = evaluate_attack(
                model, test_loader, device, attack_type='pgd',
                epsilon=eps, alpha=PGD_ALPHA, num_steps=PGD_STEPS_EVAL
            )
            fgsm_accs.append(fgsm_acc)
            pgd_accs.append(pgd_acc)
            print(f"FGSM: {fgsm_acc:.2f}%, PGD: {pgd_acc:.2f}%")
    
    return {
        'epsilons': epsilons,
        'clean_accuracy': clean_acc,
        'fgsm_accuracies': fgsm_accs,
        'pgd_accuracies': pgd_accs
    }


def evaluate():
    MODEL_NAME = "mobilenet_v2"
    IMAGE_SIZE = 160
    BATCH_SIZE = 16
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get dataloaders
    print(f"Loading dataset with image_size={IMAGE_SIZE}...")
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    
    # Get model and load weights
    print(f"Loading {MODEL_NAME} model from {MOBILENET_CLEAN_PATH}...")
    model = get_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(MOBILENET_CLEAN_PATH):
        print(f"Error: Model not found at {MOBILENET_CLEAN_PATH}. Please train it first.")
        return
        
    checkpoint = torch.load(MOBILENET_CLEAN_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 1. Clean Accuracy and Predictions
    print("\nEvaluating clean accuracy...")
    clean_labels, clean_preds, _ = get_predictions(model, test_loader, device)
    clean_acc = 100.0 * np.mean(np.array(clean_preds) == np.array(clean_labels))
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # Plot clean confusion matrix
    plot_confusion_matrix(
        clean_labels, clean_preds,
        MOBILENET_CONFUSION_CLEAN,
        f"{MODEL_NAME.upper()} - Clean Confusion Matrix"
    )
    
    # 2. FGSM Attack
    print(f"\nEvaluating FGSM robustness (eps={EPS_FGSM:.4f})...")
    fgsm_acc, _, _ = evaluate_attack(
        model, test_loader, device, 
        attack_type='fgsm', 
        epsilon=EPS_FGSM
    )
    print(f"FGSM Robust Accuracy: {fgsm_acc:.2f}%")
    
    # 3. PGD Attack and Predictions
    print(f"\nEvaluating PGD robustness (eps={EPS_PGD:.4f}, steps={PGD_STEPS_EVAL})...")
    pgd_acc, pgd_preds_tensor, pgd_labels_tensor = evaluate_attack(
        model, test_loader, device, 
        attack_type='pgd', 
        epsilon=EPS_PGD,
        alpha=PGD_ALPHA,
        num_steps=PGD_STEPS_EVAL
    )
    print(f"PGD Robust Accuracy: {pgd_acc:.2f}%")
    
    # Plot PGD confusion matrix
    plot_confusion_matrix(
        pgd_labels_tensor.tolist(), pgd_preds_tensor.tolist(),
        MOBILENET_CONFUSION_PGD,
        f"{MODEL_NAME.upper()} - PGD Attack Confusion Matrix"
    )
    
    # 4. Accuracy vs Epsilon Curve
    eps_metrics = evaluate_accuracy_vs_epsilon(
        model, test_loader, device, 
        ACCURACY_VS_EPS_EPSILONS, 
        MODEL_NAME
    )
    
    # Plot accuracy vs epsilon
    plot_accuracy_vs_epsilon(
        eps_metrics['epsilons'],
        eps_metrics['clean_accuracy'],
        eps_metrics['fgsm_accuracies'],
        eps_metrics['pgd_accuracies'],
        MOBILENET_ACC_VS_EPS
    )
    
    # Load existing metrics if any
    metrics = {}
    if os.path.exists(MOBILENET_METRICS_FILE):
        with open(MOBILENET_METRICS_FILE, 'r') as f:
            metrics = json.load(f)
            
    # Update metrics
    metrics.update({
        'clean_accuracy': clean_acc,
        'fgsm_accuracy': fgsm_acc,
        'pgd_accuracy': pgd_acc,
        'epsilons': eps_metrics['epsilons'],
        'fgsm_accuracies': eps_metrics['fgsm_accuracies'],
        'pgd_accuracies': eps_metrics['pgd_accuracies'],
        'attack_params': {
            'eps_fgsm': EPS_FGSM,
            'eps_pgd': EPS_PGD,
            'pgd_steps': PGD_STEPS_EVAL
        }
    })
    
    # Save metrics
    with open(MOBILENET_METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nAll metrics and plots updated and saved.")
    print(f"Metrics: {MOBILENET_METRICS_FILE}")
    print(f"Plots: {MOBILENET_CONFUSION_CLEAN}, {MOBILENET_CONFUSION_PGD}, {MOBILENET_ACC_VS_EPS}")


if __name__ == "__main__":
    evaluate()

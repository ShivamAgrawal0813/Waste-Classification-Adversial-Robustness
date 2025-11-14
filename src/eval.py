"""
Evaluation script for clean and adversarial models.
Evaluates robustness against FGSM and PGD attacks.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import json

from src.config import (
    CLEAN_MODEL_PATH, ADV_MODEL_PATH, NUM_CLASSES, BATCH_SIZE,
    NUM_WORKERS, EPS_FGSM, EPS_PGD, PGD_ALPHA, PGD_STEPS_EVAL,
    ACCURACY_VS_EPS_EPSILONS, METRICS_FILE, OUTPUTS_DIR,
    MODEL_NAME, PRETRAINED
)
from src.dataset import get_dataloaders
from src.attacks import fgsm_attack, pgd_attack, evaluate_attack
from src.utils import (
    set_seed, get_device, calculate_accuracy, get_predictions,
    plot_confusion_matrix, plot_accuracy_vs_epsilon, save_metrics
)
from src.train import get_model


def evaluate_model_robustness(model: nn.Module, test_loader, device: torch.device,
                             model_name: str = "model") -> dict:
    """
    Evaluate model robustness against various attacks.
    
    Args:
        model: PyTorch model
        test_loader: Test dataloader
        device: Device to run on
        model_name: Name of the model (for logging)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*60}")
    
    # Clean accuracy
    print("Evaluating clean accuracy...")
    clean_acc = calculate_accuracy(model, test_loader, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # FGSM accuracy
    print(f"\nEvaluating FGSM attack (eps={EPS_FGSM:.4f})...")
    fgsm_acc, fgsm_preds, fgsm_labels = evaluate_attack(
        model, test_loader, device, attack_type='fgsm', epsilon=EPS_FGSM
    )
    print(f"FGSM Robust Accuracy: {fgsm_acc:.2f}%")
    
    # PGD accuracy
    print(f"\nEvaluating PGD attack (eps={EPS_PGD:.4f}, steps={PGD_STEPS_EVAL})...")
    pgd_acc, pgd_preds, pgd_labels = evaluate_attack(
        model, test_loader, device, attack_type='pgd',
        epsilon=EPS_PGD, alpha=PGD_ALPHA, num_steps=PGD_STEPS_EVAL
    )
    print(f"PGD Robust Accuracy: {pgd_acc:.2f}%")
    
    # Get predictions for confusion matrices
    clean_labels, clean_preds, _ = get_predictions(model, test_loader, device)
    
    metrics = {
        'clean_accuracy': clean_acc,
        'fgsm_accuracy': fgsm_acc,
        'pgd_accuracy': pgd_acc,
        'fgsm_predictions': fgsm_preds.tolist(),
        'fgsm_labels': fgsm_labels.tolist(),
        'pgd_predictions': pgd_preds.tolist(),
        'pgd_labels': pgd_labels.tolist(),
        'clean_predictions': clean_preds,
        'clean_labels': clean_labels,
    }
    
    return metrics


def evaluate_accuracy_vs_epsilon(model: nn.Module, test_loader, device: torch.device,
                                 epsilons: list, model_name: str = "model") -> dict:
    """
    Evaluate model accuracy for different epsilon values.
    
    Args:
        model: PyTorch model
        test_loader: Test dataloader
        device: Device to run on
        epsilons: List of epsilon values to test
        model_name: Name of the model
    
    Returns:
        Dictionary with accuracy values for each epsilon
    """
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


def main():
    """Main evaluation function."""
    # Set seed
    set_seed()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get test dataloader
    print("Loading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    results = {}
    
    # Evaluate clean model
    if os.path.exists(CLEAN_MODEL_PATH):
        print(f"\nLoading clean model from {CLEAN_MODEL_PATH}...")
        model_clean = get_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
        checkpoint = torch.load(CLEAN_MODEL_PATH, map_location=device)
        model_clean.load_state_dict(checkpoint['model_state_dict'])
        model_clean = model_clean.to(device)
        model_clean.eval()
        
        # Evaluate robustness
        clean_metrics = evaluate_model_robustness(model_clean, test_loader, device, "Clean Model")
        
        # Evaluate accuracy vs epsilon
        clean_eps_metrics = evaluate_accuracy_vs_epsilon(
            model_clean, test_loader, device, ACCURACY_VS_EPS_EPSILONS, "Clean Model"
        )
        
        results['clean_model'] = {
            **clean_metrics,
            **clean_eps_metrics
        }
        
        # Plot confusion matrices
        clean_labels_list = clean_metrics['clean_labels'] if isinstance(clean_metrics['clean_labels'], list) else clean_metrics['clean_labels'].tolist()
        clean_preds_list = clean_metrics['clean_predictions'] if isinstance(clean_metrics['clean_predictions'], list) else clean_metrics['clean_predictions'].tolist()
        
        plot_confusion_matrix(
            clean_labels_list,
            clean_preds_list,
            os.path.join(OUTPUTS_DIR, 'confusion_clean.png'),
            "Clean Model - Confusion Matrix"
        )
        
        pgd_labels_list = clean_metrics['pgd_labels'] if isinstance(clean_metrics['pgd_labels'], list) else clean_metrics['pgd_labels'].tolist()
        pgd_preds_list = clean_metrics['pgd_predictions'] if isinstance(clean_metrics['pgd_predictions'], list) else clean_metrics['pgd_predictions'].tolist()
        
        plot_confusion_matrix(
            pgd_labels_list,
            pgd_preds_list,
            os.path.join(OUTPUTS_DIR, 'confusion_clean_pgd.png'),
            "Clean Model - PGD Attack Confusion Matrix"
        )
        
        # Plot accuracy vs epsilon
        plot_accuracy_vs_epsilon(
            clean_eps_metrics['epsilons'],
            clean_eps_metrics['clean_accuracy'],
            clean_eps_metrics['fgsm_accuracies'],
            clean_eps_metrics['pgd_accuracies'],
            os.path.join(OUTPUTS_DIR, 'accuracy_vs_eps_clean.png')
        )
    else:
        print(f"Clean model not found at {CLEAN_MODEL_PATH}")
    
    # Evaluate adversarial model
    if os.path.exists(ADV_MODEL_PATH):
        print(f"\nLoading adversarial model from {ADV_MODEL_PATH}...")
        model_adv = get_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
        checkpoint = torch.load(ADV_MODEL_PATH, map_location=device)
        model_adv.load_state_dict(checkpoint['model_state_dict'])
        model_adv = model_adv.to(device)
        model_adv.eval()
        
        # Evaluate robustness
        adv_metrics = evaluate_model_robustness(model_adv, test_loader, device, "Adversarial Model")
        
        # Evaluate accuracy vs epsilon
        adv_eps_metrics = evaluate_accuracy_vs_epsilon(
            model_adv, test_loader, device, ACCURACY_VS_EPS_EPSILONS, "Adversarial Model"
        )
        
        results['adversarial_model'] = {
            **adv_metrics,
            **adv_eps_metrics
        }
        
        # Plot confusion matrices
        adv_clean_labels_list = adv_metrics['clean_labels'] if isinstance(adv_metrics['clean_labels'], list) else adv_metrics['clean_labels'].tolist()
        adv_clean_preds_list = adv_metrics['clean_predictions'] if isinstance(adv_metrics['clean_predictions'], list) else adv_metrics['clean_predictions'].tolist()
        
        plot_confusion_matrix(
            adv_clean_labels_list,
            adv_clean_preds_list,
            os.path.join(OUTPUTS_DIR, 'confusion_adv.png'),
            "Adversarial Model - Confusion Matrix"
        )
        
        adv_pgd_labels_list = adv_metrics['pgd_labels'] if isinstance(adv_metrics['pgd_labels'], list) else adv_metrics['pgd_labels'].tolist()
        adv_pgd_preds_list = adv_metrics['pgd_predictions'] if isinstance(adv_metrics['pgd_predictions'], list) else adv_metrics['pgd_predictions'].tolist()
        
        plot_confusion_matrix(
            adv_pgd_labels_list,
            adv_pgd_preds_list,
            os.path.join(OUTPUTS_DIR, 'confusion_adv_pgd.png'),
            "Adversarial Model - PGD Attack Confusion Matrix"
        )
        
        # Plot accuracy vs epsilon
        plot_accuracy_vs_epsilon(
            adv_eps_metrics['epsilons'],
            adv_eps_metrics['clean_accuracy'],
            adv_eps_metrics['fgsm_accuracies'],
            adv_eps_metrics['pgd_accuracies'],
            os.path.join(OUTPUTS_DIR, 'accuracy_vs_eps.png')
        )
    else:
        print(f"Adversarial model not found at {ADV_MODEL_PATH}")
    
    # Save metrics
    if results:
        # Convert torch tensors to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert tensors and numpy arrays to lists."""
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        save_metrics(results_serializable, METRICS_FILE)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        if 'clean_model' in results:
            print(f"Clean Model:")
            print(f"  Clean Accuracy: {results['clean_model']['clean_accuracy']:.2f}%")
            print(f"  FGSM Accuracy: {results['clean_model']['fgsm_accuracy']:.2f}%")
            print(f"  PGD Accuracy: {results['clean_model']['pgd_accuracy']:.2f}%")
        if 'adversarial_model' in results:
            print(f"\nAdversarial Model:")
            print(f"  Clean Accuracy: {results['adversarial_model']['clean_accuracy']:.2f}%")
            print(f"  FGSM Accuracy: {results['adversarial_model']['fgsm_accuracy']:.2f}%")
            print(f"  PGD Accuracy: {results['adversarial_model']['pgd_accuracy']:.2f}%")
        print("="*60)
    else:
        print("No models found for evaluation.")


if __name__ == "__main__":
    main()


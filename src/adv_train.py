"""
Adversarial training script using PGD (Madry-style) for robust ResNet model.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import time

from src.config import (
    EPOCHS_ADV, LR, MOMENTUM, WEIGHT_DECAY, SCHEDULER_TYPE,
    STEP_SIZE, GAMMA, MODEL_NAME, PRETRAINED, ADV_MODEL_PATH,
    CLEAN_MODEL_PATH, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    EPS_PGD, PGD_ALPHA, PGD_STEPS_TRAIN, RANDOM_START,
    ADV_TRAIN_MIX_RATIO
)
from src.dataset import get_dataloaders
from src.attacks import pgd_attack_training
from src.utils import (
    set_seed, get_device, save_model, calculate_accuracy
)
from src.train import get_model


def train_epoch_adv(model: nn.Module, train_loader, device: torch.device,
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   mix_ratio: float = ADV_TRAIN_MIX_RATIO) -> tuple:
    """
    Train for one epoch with adversarial training.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        device: Device to run on
        criterion: Loss function
        optimizer: Optimizer
        mix_ratio: Ratio for mixing clean and adversarial loss
    
    Returns:
        Tuple of (average_loss, clean_accuracy, adv_accuracy)
    """
    model.train()
    running_loss = 0.0
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Clean forward pass
        optimizer.zero_grad()
        clean_outputs = model(images)
        clean_loss = criterion(clean_outputs, labels)
        
        # Generate adversarial examples
        adv_images = pgd_attack_training(
            model, images, labels,
            epsilon=EPS_PGD,
            alpha=PGD_ALPHA,
            num_steps=PGD_STEPS_TRAIN,
            random_start=RANDOM_START
        )
        
        # Adversarial forward pass
        adv_outputs = model(adv_images)
        adv_loss = criterion(adv_outputs, labels)
        
        # Mixed loss
        loss = (1 - mix_ratio) * clean_loss + mix_ratio * adv_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, clean_predicted = torch.max(clean_outputs.data, 1)
        _, adv_predicted = torch.max(adv_outputs.data, 1)
        total += labels.size(0)
        clean_correct += (clean_predicted == labels).sum().item()
        adv_correct += (adv_predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    return epoch_loss, clean_acc, adv_acc


def train():
    """Main adversarial training function."""
    # Set seed
    set_seed()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Load clean model or initialize new model
    print(f"Initializing {MODEL_NAME} model...")
    model = get_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
    
    # Try to load clean model weights if available
    if os.path.exists(CLEAN_MODEL_PATH):
        print(f"Loading pretrained clean model from {CLEAN_MODEL_PATH}...")
        checkpoint = torch.load(CLEAN_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Clean model weights loaded successfully.")
    else:
        print("No clean model found. Starting from scratch.")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    if SCHEDULER_TYPE == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_ADV)
    else:
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_clean_accs = []
    train_adv_accs = []
    val_accs = []
    
    print(f"\nStarting adversarial training for {EPOCHS_ADV} epochs...")
    print(f"Attack parameters: eps={EPS_PGD:.4f}, alpha={PGD_ALPHA:.4f}, steps={PGD_STEPS_TRAIN}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS_ADV):
        epoch_start = time.time()
        
        # Train with adversarial examples
        train_loss, train_clean_acc, train_adv_acc = train_epoch_adv(
            model, train_loader, device, criterion, optimizer, ADV_TRAIN_MIX_RATIO
        )
        
        # Validate on clean data
        val_acc = calculate_accuracy(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save statistics
        train_losses.append(train_loss)
        train_clean_accs.append(train_clean_acc)
        train_adv_accs.append(train_adv_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, ADV_MODEL_PATH, epoch=epoch, optimizer=optimizer,
                      metrics={
                          'val_acc': val_acc,
                          'train_clean_acc': train_clean_acc,
                          'train_adv_acc': train_adv_acc
                      })
            print(f"✓ New best model saved (val_acc: {val_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{EPOCHS_ADV}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Clean Acc: {train_clean_acc:.2f}% | "
              f"Train Adv Acc: {train_adv_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Adversarial training completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_acc = calculate_accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save final metrics
    final_metrics = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_clean_accs': train_clean_accs,
        'train_adv_accs': train_adv_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
    }
    save_model(model, ADV_MODEL_PATH, epoch=EPOCHS_ADV-1,
              optimizer=optimizer, metrics=final_metrics)
    
    print(f"\nAdversarially trained model saved to {ADV_MODEL_PATH}")


if __name__ == "__main__":
    train()


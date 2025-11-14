"""
Training script for clean (non-adversarial) ResNet model on TrashNet.
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
    EPOCHS_CLEAN, LR, MOMENTUM, WEIGHT_DECAY, SCHEDULER_TYPE,
    STEP_SIZE, GAMMA, MODEL_NAME, PRETRAINED, CLEAN_MODEL_PATH,
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS
)
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, get_device, save_model, load_model, calculate_accuracy
)
import torchvision.models as models


def get_model(model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES,
              pretrained: bool = PRETRAINED) -> nn.Module:
    """
    Get ResNet model.
    
    Args:
        model_name: 'resnet50' or 'resnet18'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        ResNet model
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model: nn.Module, train_loader, device: torch.device,
               criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train():
    """Main training function."""
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
    
    # Get model
    print(f"Initializing {MODEL_NAME} model...")
    model = get_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
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
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_CLEAN)
    else:
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\nStarting training for {EPOCHS_CLEAN} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS_CLEAN):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, device, criterion, optimizer)
        
        # Validate
        val_acc = calculate_accuracy(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save statistics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, CLEAN_MODEL_PATH, epoch=epoch, optimizer=optimizer,
                      metrics={'val_acc': val_acc, 'train_acc': train_acc})
            print(f"✓ New best model saved (val_acc: {val_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{EPOCHS_CLEAN}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_acc = calculate_accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save final metrics
    final_metrics = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
    }
    save_model(model, CLEAN_MODEL_PATH, epoch=EPOCHS_CLEAN-1, 
              optimizer=optimizer, metrics=final_metrics)
    
    print(f"\nModel saved to {CLEAN_MODEL_PATH}")


if __name__ == "__main__":
    train()


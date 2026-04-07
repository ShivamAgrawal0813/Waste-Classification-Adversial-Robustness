"""
Training script for MobileNet-V2 clean model on TrashNet.
Custom parameters: image_size=160, batch_size=16, epochs=40, num_workers=2.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from src.config import (
    LR, MOMENTUM, WEIGHT_DECAY, NUM_CLASSES, MOBILENET_CLEAN_PATH, 
    MOBILENET_METRICS_FILE
)
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, get_device, save_model, calculate_accuracy
)
from src.train import get_model, train_epoch


def train():
    # Parameters requested by user
    MODEL_NAME = "mobilenet_v2"
    IMAGE_SIZE = 160
    BATCH_SIZE = 16
    EPOCHS = 40
    NUM_WORKERS = 2
    
    # Set seed
    set_seed()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get dataloaders
    print(f"Loading dataset with image_size={IMAGE_SIZE}, batch_size={BATCH_SIZE}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    # Get model
    print(f"Initializing {MODEL_NAME} model (Transfer Learning)...")
    model = get_model(MODEL_NAME, NUM_CLASSES, pretrained=True)
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
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
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
            save_model(model, MOBILENET_CLEAN_PATH, epoch=epoch, optimizer=optimizer,
                      metrics={'val_acc': val_acc, 'train_acc': train_acc})
            print(f"✓ New best model saved (val_acc: {val_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_acc = calculate_accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save final metrics
    import json
    final_metrics = {
        'model_name': MODEL_NAME,
        'image_size': IMAGE_SIZE,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
    }
    
    with open(MOBILENET_METRICS_FILE, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"\nModel saved to {MOBILENET_CLEAN_PATH}")
    print(f"Metrics saved to {MOBILENET_METRICS_FILE}")


if __name__ == "__main__":
    train()

"""
Dataset loader for TrashNet.
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List, Optional

from src.config import (
    DATASET_PATH, CLASS_TO_IDX, NUM_CLASSES, IMAGE_SIZE, 
    RESIZE_SIZE, BATCH_SIZE, NUM_WORKERS, MEAN, STD,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED
)
from src.utils import set_seed


class TrashNetDataset(Dataset):
    """TrashNet dataset loader."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get data transforms for training or testing.
    
    Args:
        mode: 'train' or 'test'
    
    Returns:
        Compose transform
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
        ])
    else:  # test/val
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


def load_dataset() -> Tuple[List[str], List[int]]:
    """
    Load all images and labels from the TrashNet dataset.
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Try to find the dataset in different possible locations
    base_dir = os.path.dirname(os.path.dirname(DATASET_PATH))  # Go up from config path
    possible_paths = [
        DATASET_PATH,
        os.path.join(base_dir, "data", "trashnet", "Garbage classification", "Garbage classification"),
        os.path.join(base_dir, "data", "trashnet", "trashnet", "trashnet"),
        os.path.join(base_dir, "data", "trashnet"),
        os.path.join(os.path.dirname(DATASET_PATH), "Garbage classification", "Garbage classification"),
        os.path.join(os.path.dirname(DATASET_PATH), "trashnet", "trashnet"),
    ]
    
    dataset_root = None
    for path in possible_paths:
        if os.path.exists(path):
            # Check if this path contains class folders
            has_classes = any(os.path.exists(os.path.join(path, cls)) for cls in CLASS_TO_IDX.keys())
            if has_classes:
                dataset_root = path
                break
    
    if dataset_root is None:
        # Last attempt: search for class folders in subdirectories
        for path in possible_paths:
            if os.path.exists(path):
                # Search recursively for class folders
                for root, dirs, files in os.walk(path):
                    if any(cls in dirs for cls in CLASS_TO_IDX.keys()):
                        # Check if at least 3 classes are present
                        found_classes = [cls for cls in CLASS_TO_IDX.keys() if cls in dirs]
                        if len(found_classes) >= 3:
                            dataset_root = root
                            break
                if dataset_root:
                    break
    
    if dataset_root is None:
        raise FileNotFoundError(
            f"Dataset not found. Tried: {possible_paths}\n"
            f"Please ensure TrashNet dataset is in data/trashnet/\n"
            f"Expected structure: data/trashnet/.../cardboard/, glass/, metal/, paper/, plastic/, trash/"
        )
    
    print(f"Found dataset at: {dataset_root}")
    
    # Load images from each class folder
    for class_name in CLASS_TO_IDX.keys():
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.exists(class_dir):
            # Try to find class folder in subdirectories
            for root, dirs, files in os.walk(dataset_root):
                if class_name in dirs:
                    class_dir = os.path.join(root, class_name)
                    break
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_name}")
            continue
        
        class_idx = CLASS_TO_IDX[class_name]
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} images from {len(set(labels))} classes")
    return image_paths, labels


def split_dataset(image_paths: List[str], labels: List[int], 
                 train_split: float = TRAIN_SPLIT,
                 val_split: float = VAL_SPLIT,
                 test_split: float = TEST_SPLIT,
                 seed: int = SEED) -> Tuple[List[Tuple[str, int]], 
                                           List[Tuple[str, int]], 
                                           List[Tuple[str, int]]]:
    """
    Split dataset into train, val, and test sets.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        seed: Random seed
    
    Returns:
        Tuple of (train_data, val_data, test_data) where each is a list of (path, label) tuples
    """
    set_seed(seed)
    
    # Combine paths and labels
    data = list(zip(image_paths, labels))
    
    # Shuffle
    random.shuffle(data)
    
    # Calculate split indices
    total = len(data)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def get_dataloaders(batch_size: int = BATCH_SIZE, 
                   num_workers: int = NUM_WORKERS,
                   seed: int = SEED) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders.
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    image_paths, labels = load_dataset()
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(image_paths, labels, seed=seed)
    
    # Unzip data
    train_paths, train_labels = zip(*train_data) if train_data else ([], [])
    val_paths, val_labels = zip(*val_data) if val_data else ([], [])
    test_paths, test_labels = zip(*test_data) if test_data else ([], [])
    
    # Create datasets
    train_dataset = TrashNetDataset(
        list(train_paths), 
        list(train_labels), 
        transform=get_transforms('train')
    )
    val_dataset = TrashNetDataset(
        list(val_paths), 
        list(val_labels), 
        transform=get_transforms('test')
    )
    test_dataset = TrashNetDataset(
        list(test_paths), 
        list(test_labels), 
        transform=get_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


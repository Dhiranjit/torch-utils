"""
Contains functionality for creating PyTorch datasets and dataloaders
from a standard image classification dataset structure.
"""
import torch
from pathlib import Path
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def create_dataloader(
    train_dir: str | Path,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    test_dir: str | Path | None = None, # Optional
    val_dir: str | Path | None = None,  # Optional
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
):
    """
    Creates PyTorch DataLoaders handling various directory structures.

    Scenarios:
    1. Train + Val + Test provided: Loads all strictly from folders.
    2. Train + Test (No Val): Splits Train -> (Train + Val), loads Test.
    3. Train + Val (No Test): Loads Train and Val, returns None for Test.
    4. Train Only: Splits Train -> (Train + Val), returns None for Test.

    Returns:
        (train_dataloader, val_dataloader, test_dataloader, class_names)
        *Note: test_dataloader will be None if test_dir is not provided.*
    """

    # ---------------------------------------------------------
    # PART 1: Handle Training and Validation Data
    # ---------------------------------------------------------
    
    # Case A: Validation directory is explicitly provided
    if val_dir is not None:
        train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_data = datasets.ImageFolder(root=val_dir, transform=test_transform)
        class_names = train_data.classes

    # Case B: No validation directory -> Split TRAIN into train and val
    else:
        # Create two dataset objects over the SAME files but with different transforms
        full_data_train_tf = datasets.ImageFolder(root=train_dir, transform=train_transform)
        full_data_val_tf = datasets.ImageFolder(root=train_dir, transform=test_transform)
        
        class_names = full_data_train_tf.classes
        
        total_len = len(full_data_train_tf)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        
        # Generate random permutation of indices
        indices = torch.randperm(total_len).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        
        # Create Subsets
        train_data = Subset(full_data_train_tf, train_indices)
        val_data   = Subset(full_data_val_tf, val_indices)

    # ---------------------------------------------------------
    # PART 2: Handle Test Data (Optional)
    # ---------------------------------------------------------
    
    test_dataloader = None
    
    if test_dir is not None:
        test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # ---------------------------------------------------------
    # PART 3: Create Train/Val Loaders
    # ---------------------------------------------------------

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
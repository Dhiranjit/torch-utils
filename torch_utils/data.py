"""
Contains functionality for creating PyTorch datasets and dataloaders
from a standard image classification dataset structure.
"""
import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def create_dataloader(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 0,
    val_dir: str | None = None,
    val_split: float = 0.2,
):
    """
    Creates PyTorch DataLoaders from a standard image classification directory structure.

    Behavior:
    - If val_dir is provided:
        train_dir -> train
        val_dir   -> validation
        test_dir  -> test
    - If val_dir is None:
        train_dir -> split into (train, validation)
        test_dir  -> test (untouched)

    Validation and test data always use test_transform.
    Training data uses train_transform.

    Returns:
        (train_dataloader, val_dataloader, test_dataloader, class_names)
    """

    # Case 1: Validation directory is explicitly provided
    if val_dir is not None:
        train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_data = datasets.ImageFolder(root=val_dir, transform=test_transform)
        class_names = train_data.classes

    # Case 2: No validation directory → split TRAIN into train and val
    else:
        # Create two dataset objects over the SAME files but with different transforms
        full_data_train_tf = datasets.ImageFolder(root=train_dir, transform=train_transform)
        full_data_val_tf = datasets.ImageFolder(root=train_dir, transform=test_transform)

        class_names = full_data_train_tf.classes

        total_len = len(full_data_train_tf)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        
        # We generate a random permutation of indices once (Prevent Data Leakage)
        indices = torch.randperm(total_len).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        
        # 4. Create Subsets using the SAME indices for both transform versions
        train_data = Subset(full_data_train_tf, train_indices)
        val_data   = Subset(full_data_val_tf, val_indices)

    # Test dataset is NEVER split and always uses test_transform
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Create DataLoaders
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

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names

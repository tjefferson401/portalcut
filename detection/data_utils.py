from datasets import KittiTorch
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import utils


from datasets import KittiTorch


def get_dataloaders(dataset):
    indices = torch.randperm(len(dataset)).tolist()

    # Calculate split sizes, it should add up to 1
    train_split = 0.8
    val_split = 0.1  # 10% for validation
    test_split = 0.1  # 10% for test

    # Calculate indices for splits
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size  # To ensure full coverage

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create dataset subsets
    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)
    dataset_test = Subset(dataset, test_indices)

    # Define batch size
    batch_size = 4

    # Data loaders
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # Print sizes of datasets to confirm splits
    print("Training set size:", len(dataset_train))
    print("Validation set size:", len(dataset_val))
    print("Testing set size:", len(dataset_test))
    
    return train_dataloader, val_dataloader, test_dataloader

def get_transform():
    transform = [transforms.ToTensor()]
    return transforms.Compose(transform)
  




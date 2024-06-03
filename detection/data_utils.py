import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import utils




def get_datamoduels(dataset, batch_size=4):
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
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # Print sizes of datasets to confirm splits
    print("Dataset size:", len(dataset))
    print(f"Batch size: {batch_size}")
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Testing set size:", len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

def get_transform():
    transform = [transforms.ToTensor()]
    return transforms.Compose(transform)
  




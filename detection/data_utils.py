import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
import utils


def get_transform():
    transform = [transforms.ToTensor()]
    return transforms.Compose(transform)


def generate_splits(dataset, batch_size=4):
    indices = torch.randperm(len(dataset)).tolist()

    # Calculate split sizes
    train_split = 0.8
    val_split = 0.1  # 10% for validation
    test_split = 0.1  # 10% for test

    # Calculate exact indices for splits
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size  # Ensure full coverage

    return indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]

def create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, augmented_indices=None):
    if augmented_indices is not None:
        train_indices += augmented_indices  # Add augmented indices only to the training set

    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

def get_datamodules(dataset, batch_size=4):
    train_indices, val_indices, test_indices = generate_splits(dataset, batch_size)
    return create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size)

def get_augmented_datamodules(dataset, batch_size=4):
    train_indices, val_indices, test_indices = generate_splits(dataset, batch_size)
    augmented_indices = [i for i, filename in enumerate(dataset.filepaths) if '_' in filename.split('/')[-1]]
    return create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, augmented_indices)


def create_reduced_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, original_data_percentage, augmented_indices=None):
    # Reduce the number of original training indices based on the percentage specified
    reduced_train_size = int(len(train_indices) * original_data_percentage)
    train_indices = train_indices[:reduced_train_size]
    
    if augmented_indices is not None:
        train_indices += augmented_indices  # Add augmented indices only to the training set

    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

def get_reduced_datamodules(dataset, batch_size=4, original_data_percentage=1.0):
    train_indices, val_indices, test_indices = generate_splits(dataset, batch_size)
    return create_reduced_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, original_data_percentage)

def get_augmented_reduced_datamodules(dataset, batch_size=4, original_data_percentage=1.0):
    train_indices, val_indices, test_indices = generate_splits(dataset, batch_size)
    augmented_indices = [i for i, filename in enumerate(dataset.filepaths) if '_' in filename.split('/')[-1]]
    return create_reduced_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, original_data_percentage, augmented_indices)

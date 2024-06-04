import torch
from torch.utils.data import DataLoader, Subset, random_split
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




def get_augmented_datamodules(dataset, batch_size=4):
    augmented_indices = []
    original_indices = []

    # Iterate over all dataset entries to classify them based on filename
    for idx in range(len(dataset)):
        filename = dataset.get_filename(idx)
        # Isolate the actual filename from the full path
        actual_filename = filename.split('/')[-1]
        # Check if the actual filename contains an underscore (specific to augmented files)
        if '_' in actual_filename:  # Adjust the condition based on your naming convention
            augmented_indices.append(idx)
        else:
            original_indices.append(idx)

    # Shuffle original indices to randomly select for train, val, test
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    original_indices = torch.tensor(original_indices)
    shuffled_original_indices = torch.randperm(len(original_indices), generator=generator).tolist()
    
    # Calculate the sizes of original splits
    num_original = len(original_indices)
    train_size = int(num_original * 0.8)
    val_size = int(num_original * 0.1)
    test_size = num_original - train_size - val_size

    # Split original indices
    original_train_indices = original_indices[shuffled_original_indices[:train_size]]
    original_val_indices = original_indices[shuffled_original_indices[train_size:train_size + val_size]]
    original_test_indices = original_indices[shuffled_original_indices[train_size + val_size:]]

    # Combine augmented indices with the training indices from original
    train_indices = torch.cat((original_train_indices, torch.tensor(augmented_indices)))

    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, original_val_indices.tolist())
    test_dataset = Subset(dataset, original_test_indices.tolist())

    # Data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )
    
    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader




def get_transform():
    transform = [transforms.ToTensor()]
    return transforms.Compose(transform)
  




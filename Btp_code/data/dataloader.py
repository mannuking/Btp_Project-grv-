import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Sort files to ensure consistent order
        self.image_files = sorted(self.image_files)
        
        # Label assignment based on index: first 500 are fake (1), remaining are real (0)
        # This overrides the filename-based approach
        self.labels = [1 if i < 500 else 0 for i in range(len(self.image_files))]
        
        print(f"Loaded dataset with {len(self.image_files)} images")
        print(f"Real images: {self.labels.count(0)}, Fake images: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloader(data_dir, batch_size, shuffle=True, transform=None, split='train', test_ratio=0.2, seed=42):
    """
    Create dataloader with proper train/test splitting.
    
    Args:
        data_dir: Directory containing image files
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the data
        transform: Transformations to apply to images
        split: 'train', 'test', or 'all' to return a specific split
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
    """
    # Define transforms if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    # Create the full dataset
    full_dataset = DeepfakeDataset(data_dir, transform=transform)
    
    # If we want all the data without splitting
    if split == 'all':
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    # Calculate the split sizes
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - test_size
    
    # Create the splits using random_split
    torch.manual_seed(seed)  # For reproducibility
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Return the requested split
    if split == 'train':
        print(f"Created training dataloader with {len(train_dataset)} images")
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    elif split == 'test':
        print(f"Created test dataloader with {len(test_dataset)} images")
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        raise ValueError(f"Invalid split type: {split}. Must be 'train', 'test', or 'all'")

# Example Usage (assuming data is in 'data/processed')
if __name__ == '__main__':
    from utils.config import config

    # Define transforms (example)
    data_transforms = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    # Create dataloader
    # Note: You'll need to have some images in 'data/processed' for this to run
    try:
        # Create train and test dataloaders
        train_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, transform=data_transforms, split='train')
        test_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, transform=data_transforms, split='test')
        
        print(f"Train dataloader created with {len(train_dataloader.dataset)} images")
        print(f"Test dataloader created with {len(test_dataloader.dataset)} images")
        
    except FileNotFoundError:
        print(f"Error: Data directory not found at {config.PROCESSED_DIR}. Please add some images.") 

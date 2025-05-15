import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        # Assuming you have a way to determine labels (real/fake) based on file names or a separate file
        # For now, we'll just create dummy labels
        self.labels = [0] * len(self.image_files) # 0 for real, 1 for fake (placeholder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloader(data_dir, batch_size, shuffle=True, transform=None):
    dataset = DeepfakeDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

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
        dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, transform=data_transforms)
        print(f"Created dataloader with {len(dataloader.dataset)} images.")
        
        # Example of iterating through the dataloader
        # for images, labels in dataloader:
        #     print(f"Batch of images shape: {images.shape}")
        #     print(f"Batch of labels shape: {labels.shape}")
        #     break

    except FileNotFoundError:
        print(f"Error: Data directory not found at {config.PROCESSED_DIR}. Please add some images.") 
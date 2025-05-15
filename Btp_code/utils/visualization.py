import matplotlib.pyplot as plt
import numpy as np
import torch

def show_images(images, titles=None, cmap='viridis', save_path=None):
    """Displays a grid of images."""
    if isinstance(images, torch.Tensor):
        # Convert torch tensor to numpy array and rearrange dimensions for plotting (CxHxW -> HxWxC)
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        # Denormalize if necessary (assuming ImageNet normalization)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # images = images * std + mean
        images = np.clip(images, 0, 1) # Clip to [0, 1] after potential denormalization

    num_images = len(images)
    fig = plt.figure(figsize=(10, 10))
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1)
        ax.imshow(images[i], cmap=cmap)
        if titles:
            ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example Usage
if __name__ == '__main__':
    # Dummy image data (example: 3 images, 3 channels, 64x64)
    dummy_images_np = np.random.rand(3, 64, 64, 3) * 255 # Example numpy image (HxWx channels)
    dummy_images_np = dummy_images_np.astype(np.uint8)
    dummy_titles = ['Image 1', 'Image 2', 'Image 3']

    # Show numpy images
    print("Showing numpy images:")
    # show_images(dummy_images_np, titles=dummy_titles)

    # Dummy torch tensor image data (example: 3 images, 3 channels, 64x64)
    dummy_images_torch = torch.randn(3, 3, 64, 64) # Example torch tensor (BxCxHxW)
    # Assume these are normalized, so no denormalization needed for this example

    # Show torch tensor images
    print("Showing torch tensor images:")
    # show_images(dummy_images_torch, titles=dummy_titles, cmap=None) # Use cmap=None for color images

    # Dummy loss data
    dummy_train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    dummy_val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]

    # Plot loss curves
    print("Plotting loss curves:")
    # plot_loss_curves(dummy_train_losses, dummy_val_losses) 
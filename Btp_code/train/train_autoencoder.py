import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.autoencoder import Autoencoder
from data.dataloader import create_dataloader
from utils.config import config
from utils.logger import setup_logger, save_checkpoint
from utils.visualization import show_images # Might be useful for visualizing reconstructions

def train_autoencoder():
    # Setup logger
    log_file = os.path.join(config.LOG_DIR, 'train_autoencoder.log')
    logger = setup_logger('Autoencoder_Trainer', log_file)

    logger.info("Starting Autoencoder training...")

    # Device configuration
    device = torch.device(config.DEVICE)

    # Initialize Autoencoder
    img_shape = (config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    autoencoder = Autoencoder(img_shape, config.LATENT_DIM).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = optim.Adam(autoencoder.parameters(), lr=config.LEARNING_RATE)

    # Create dataloader
    dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0.0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            reconstructed_images = autoencoder(images)
            loss = criterion(reconstructed_images, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log progress
            if (i + 1) % 100 == 0: # Log every 100 batches
                logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")

        # Visualize reconstructions (optional)
        if (epoch + 1) % 10 == 0: # Visualize every 10 epochs
            with torch.no_grad():
                # Take a batch of images from the dataloader
                dataiter = iter(dataloader)
                sample_images, _ = next(dataiter)
                sample_images = sample_images[:8].to(device) # Take first 8 images
                reconstructed_samples = autoencoder(sample_images).cpu()
                # show_images(sample_images.cpu(), titles=['Original']*sample_images.size(0), cmap=None) # Show original images
                # show_images(reconstructed_samples, titles=['Reconstructed']*reconstructed_samples.size(0), cmap=None, save_path=os.path.join(config.RESULT_DIR, f'autoencoder_reconstructed_epoch_{epoch+1}.png')) # Show reconstructions

        # Save checkpoint (optional)
        if (epoch + 1) % 50 == 0: # Save every 50 epochs
             save_checkpoint(autoencoder, optimizer, epoch, avg_loss, config.CHECKPOINT_DIR, 'autoencoder')

    logger.info("Autoencoder training finished.")

if __name__ == '__main__':
    train_autoencoder() 
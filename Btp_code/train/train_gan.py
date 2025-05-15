import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os

from models.gan import Generator, Discriminator
from data.dataloader import create_dataloader
from utils.config import config
from utils.logger import setup_logger, save_checkpoint
from utils.visualization import show_images # Might be useful for visualizing generated images

def train_gan():
    # Setup logger
    log_file = os.path.join(config.LOG_DIR, 'train_gan.log')
    logger = setup_logger('GAN_Trainer', log_file)

    logger.info("Starting GAN training...")

    # Device configuration
    device = torch.device(config.DEVICE)

    # Initialize Generator and Discriminator
    img_shape = (config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    generator = Generator(config.LATENT_DIM, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Create dataloader
    # Assuming data is preprocessed and available in config.PROCESSED_DIR
    # You might need a separate dataloader for real images depending on your dataset structure
    dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        for i, (real_images, _) in enumerate(dataloader):
            # Assuming dataloader provides real images and dummy labels
            real_images = real_images.to(device)

            # Adversarial ground truths
            valid = torch.ones(real_images.size(0), 1, device=device)
            fake = torch.zeros(real_images.size(0), 1, device=device)

            # -----------------
            # Train Discriminator
            # -----------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real images
            real_loss = adversarial_loss(discriminator(real_images), valid)

            # Generate fake images
            z = torch.randn(real_images.size(0), config.LATENT_DIM, device=device)
            fake_images = generator(z).detach() # Detach to not update generator

            # Measure discriminator's ability to classify fake images
            fake_loss = adversarial_loss(discriminator(fake_images), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate fake images (this time update generator)
            gen_images = generator(z)

            # Measure generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_images), valid)

            g_loss.backward()
            optimizer_G.step()

            # --------------
            # Log Progress
            # --------------

            if (i + 1) % 100 == 0: # Log every 100 batches
                logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        # Save generated images for visualization (optional)
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
            with torch.no_grad():
                # Generate a fixed set of images to track progress
                fixed_noise = torch.randn(16, config.LATENT_DIM, device=device)
                generated_samples = generator(fixed_noise).cpu()
                save_image(generated_samples, os.path.join(config.RESULT_DIR, f'gan_generated_epoch_{epoch+1}.png'), nrow=4, normalize=True)

        # Save checkpoint (optional)
        if (epoch + 1) % 50 == 0: # Save every 50 epochs
             save_checkpoint(generator, optimizer_G, epoch, g_loss.item(), config.CHECKPOINT_DIR, 'gan_generator')
             save_checkpoint(discriminator, optimizer_D, epoch, d_loss.item(), config.CHECKPOINT_DIR, 'gan_discriminator')

    logger.info("GAN training finished.")

if __name__ == '__main__':
    train_gan() 
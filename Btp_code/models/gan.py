import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))), # Flattened image output
            nn.Tanh()
        )

    def forward(self, z):
        # z is a random noise vector (latent_dim)
        img = self.model(z)
        # Reshape the output to be an image (batch_size, channels, height, width)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), # Output a single value (real/fake score)
            nn.Sigmoid() # For binary classification
        )

    def forward(self, img):
        # Flatten the image
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Define input shape based on config
    img_shape = (config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    latent_dim = config.LATENT_DIM

    # Instantiate models
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape)

    # Create a dummy latent vector
    dummy_latent_vector = torch.randn(config.BATCH_SIZE, latent_dim)

    # Generate a fake image
    fake_images = generator(dummy_latent_vector)
    print(f"Shape of generated fake images: {fake_images.shape}")

    # Pass fake images through the discriminator
    fake_validity = discriminator(fake_images)
    print(f"Shape of discriminator output for fake images: {fake_validity.shape}")

    # Create a dummy real image batch (assuming same shape as generated)
    dummy_real_images = torch.randn(config.BATCH_SIZE, *img_shape)

    # Pass real images through the discriminator
    real_validity = discriminator(dummy_real_images)
    print(f"Shape of discriminator output for real images: {real_validity.shape}") 
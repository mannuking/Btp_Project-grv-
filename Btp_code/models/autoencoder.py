import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        img_features = int(torch.prod(torch.tensor(img_shape)))

        self.model = nn.Sequential(
            nn.Linear(img_features, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim) # Output latent vector
        )

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        z = self.model(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        img_features = int(torch.prod(torch.tensor(img_shape)))

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_features),
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def forward(self, z):
        # z is the latent vector
        img_flat = self.model(z)
        # Reshape the output to be an image
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img

class Autoencoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(img_shape, latent_dim)
        self.decoder = Decoder(latent_dim, img_shape)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Define input shape based on config
    img_shape = (config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    latent_dim = config.LATENT_DIM

    # Instantiate Autoencoder
    autoencoder = Autoencoder(img_shape, latent_dim)

    # Create a dummy image batch
    dummy_images = torch.randn(config.BATCH_SIZE, *img_shape)

    # Pass images through the autoencoder
    reconstructed_images = autoencoder(dummy_images)
    print(f"Shape of reconstructed images: {reconstructed_images.shape}")

    # Example of using Encoder and Decoder separately
    # encoder = Encoder(img_shape, latent_dim)
    # decoder = Decoder(latent_dim, img_shape)
    # latent_representation = encoder(dummy_images)
    # print(f"Shape of latent representation: {latent_representation.shape}")
    # decoded_images = decoder(latent_representation)
    # print(f"Shape of decoded images: {decoded_images.shape}") 
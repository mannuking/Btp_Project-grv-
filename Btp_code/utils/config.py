import os
import torch

class Config:
    # Global settings
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001

    # Data settings
    DATA_DIR = 'data/celebA'
    PROCESSED_DIR = 'data/processed'
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 3
    TRAIN_SPLIT = 0.8

    # Model settings
    LATENT_DIM = 100 # For GANs/Autoencoders

    # Output settings
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    RESULT_DIR = 'results'

    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = Config() 
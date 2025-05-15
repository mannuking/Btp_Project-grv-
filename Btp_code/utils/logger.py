import logging
import os
import torch

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Add a console handler as well
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, model_name):
    """Saves the model and optimizer state."""
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Loads the model and optimizer state from a checkpoint."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Checkpoint loaded. Resuming from epoch {epoch+1}")
    return model, optimizer, epoch, loss

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Setup logger
    log_file_path = os.path.join(config.LOG_DIR, 'test.log')
    logger = setup_logger('TestLogger', log_file_path)

    logger.info("Logger setup successfully.")

    # Dummy model and optimizer for checkpointing example
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    dummy_epoch = 5
    dummy_loss = 0.1

    # Save checkpoint (ensure config.CHECKPOINT_DIR exists)
    # save_checkpoint(dummy_model, dummy_optimizer, dummy_epoch, dummy_loss, config.CHECKPOINT_DIR, 'dummy_model')

    # Example of loading a checkpoint (you would need a saved file first)
    # try:
    #     loaded_model, loaded_optimizer, last_epoch, last_loss = load_checkpoint(DummyModel(), torch.optim.Adam(DummyModel().parameters(), lr=0.001), os.path.join(config.CHECKPOINT_DIR, 'dummy_model_epoch_6.pth'), config.DEVICE)
    #     logger.info(f"Loaded checkpoint from epoch {last_epoch+1} with loss {last_loss}")
    # except FileNotFoundError:
    #     logger.warning("Dummy checkpoint file not found.") 
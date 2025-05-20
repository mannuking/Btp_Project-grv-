import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.resnet import ResNet
from data.dataloader import create_dataloader
from utils.config import config
from utils.logger import setup_logger, save_checkpoint
from utils.metrics import calculate_metrics
from utils.visualization import plot_loss_curves # Might be useful for plotting loss

def train_resnet():
    # Setup logger
    log_file = os.path.join(config.LOG_DIR, 'train_resnet.log')
    logger = setup_logger('ResNet_Trainer', log_file)

    logger.info("Starting ResNet training...")

    # Device configuration
    device = torch.device(config.DEVICE)

    # Initialize ResNet model (using pretrained for transfer learning)
    # Make sure the input image size in dataloader matches ResNet's expected input (typically 224x224)
    resnet_model = ResNet(num_classes=2, pretrained=True).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=config.LEARNING_RATE)

    # Create dataloaders (using train/validation split)
    train_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=True, split='train')
    val_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=False, split='test')

    # Training loop
    train_losses = []
    val_losses = [] # For validation

    for epoch in range(config.NUM_EPOCHS):
        resnet_model.train() # Set model to training mode
        total_train_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Log progress
            if (i + 1) % 100 == 0: # Log every 100 batches
                logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Batch [{i+1}/{len(train_dataloader)}] Train Loss: {loss.item():.4f}")

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Average Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        resnet_model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = resnet_model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Average Validation Loss: {avg_val_loss:.4f}")

        # Calculate and log metrics
        metrics = calculate_metrics(all_predictions, all_labels)
        logger.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Validation Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")

        # Save checkpoint (optional)
        if (epoch + 1) % 5 == 0: # Save every 5 epochs instead of 50
             save_checkpoint(resnet_model, optimizer, epoch, avg_train_loss, config.CHECKPOINT_DIR, 'resnet')

    logger.info("ResNet training finished.")

    # Plot loss curves after training (optional)
    # plot_loss_curves(train_losses, val_losses, save_path=os.path.join(config.RESULT_DIR, 'resnet_loss_curves.png'))

if __name__ == '__main__':
    train_resnet() 

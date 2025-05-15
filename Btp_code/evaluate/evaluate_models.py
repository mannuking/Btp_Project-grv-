import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from data.dataloader import create_dataloader
from utils.config import config
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
from utils.visualization import show_images # For displaying images with Grad-CAM

# Import models (assuming they are implemented in the models directory)
from models.resnet import ResNet
from models.densenet import DenseNet
from models.hybrid_model import HybridModel

# You might also need to import GAN/Autoencoder if evaluating generation quality, but evaluate_models.py seems focused on detection.
# from models.gan import Generator
# from models.autoencoder import Autoencoder

def load_model(model_name, checkpoint_path, num_classes, device):
    """Loads a trained model from a checkpoint."""
    if model_name == 'resnet':
        model = ResNet(num_classes=num_classes, pretrained=False) # Load without pre-training weights initially
    elif model_name == 'densenet':
        model = DenseNet(num_classes=num_classes, pretrained=False)
    elif model_name == 'hybrid_model':
        model = HybridModel(num_classes=num_classes, input_channels=config.CHANNELS)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

def evaluate_model(model, dataloader, device):
    """Evaluates the model on the given dataloader."""
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Assuming classification task, get predictions
            # If model outputs logits, apply softmax or sigmoid depending on the loss used during training
            # For CrossEntropyLoss, argmax on logits is appropriate
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels)

    return metrics

# Function for Grad-CAM (requires a bit more setup, will be a placeholder for now)
def apply_grad_cam(model, image, target_layer, device):
    """Applies Grad-CAM to an image (placeholder)."""
    # This requires hooking into the model's forward pass and gradients.
    # Libraries like pytorch-gradcam can simplify this.
    # For now, this is just a conceptual placeholder.
    print("Grad-CAM functionality to be implemented.")
    return None # Return None or dummy data

def evaluate_models():
    # Setup logger
    log_file = os.path.join(config.LOG_DIR, 'evaluate_models.log')
    logger = setup_logger('Model_Evaluator', log_file)

    logger.info("Starting model evaluation...")

    # Device configuration
    device = torch.device(config.DEVICE)

    # Create dataloader for evaluation (assuming a separate test set or using a split of processed data)
    # You would typically need a separate test dataloader.
    # For simplicity, this example re-uses the processed data dataloader.
    eval_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=False)

    # Define models to evaluate and their checkpoint paths
    # You will need to train the models first and save checkpoints.
    models_to_evaluate = {
        'resnet': os.path.join(config.CHECKPOINT_DIR, 'resnet_epoch_X.pth'), # Replace X with epoch number
        'densenet': os.path.join(config.CHECKPOINT_DIR, 'densenet_epoch_X.pth'),
        'hybrid_model': os.path.join(config.CHECKPOINT_DIR, 'hybrid_model_epoch_X.pth'),
    }

    results = {}

    for model_name, checkpoint_path in models_to_evaluate.items():
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for {model_name} at {checkpoint_path}. Skipping evaluation.")
            continue

        logger.info(f"Evaluating {model_name}...")
        try:
            model = load_model(model_name, checkpoint_path, num_classes=2, device=device)
            metrics = evaluate_model(model, eval_dataloader, device)
            results[model_name] = metrics
            logger.info(f"{model_name} Evaluation Metrics: Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"{model_name} Confusion Matrix:\n{metrics['confusion_matrix']}")
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    logger.info("Model evaluation finished.")
    return results

if __name__ == '__main__':
    # Before running this, ensure you have trained models and saved checkpoints
    # Also, make sure you have data in the config.PROCESSED_DIR
    evaluation_results = evaluate_models()
    print("\nEvaluation Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"--- {model_name} ---")
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print("  Confusion Matrix:")
            print(metrics['confusion_matrix'])

    # Example of using Grad-CAM (requires a specific image and target layer)
    # This part needs actual implementation and depends on the model architecture.
    # if 'resnet' in evaluation_results and 'resnet' not in evaluation_results['resnet']:
    #     # Load the trained ResNet model
    #     # You would need to select a specific image from your dataset
    #     # And identify a suitable target layer for Grad-CAM (e.g., the last convolutional layer)
    #     # dummy_image = torch.randn(1, 3, 224, 224).to(device) # Example dummy image
    #     # resnet_model_eval = load_model('resnet', models_to_evaluate['resnet'], num_classes=2, device=device)
    #     # target_layer = resnet_model_eval.resnet.layer4[-1] # Example target layer for ResNet50
    #     # grad_cam_result = apply_grad_cam(resnet_model_eval, dummy_image, target_layer, device)
    #     # if grad_cam_result:
    #     #     # Visualize the Grad-CAM heatmap and overlaid image
    #     #     pass # Implementation needed 
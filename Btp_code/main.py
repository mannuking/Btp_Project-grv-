import argparse
import os
import cv2
import glob
import logging
from pathlib import Path
import shutil

# Import functions for training and evaluation
from train.train_gan import train_gan
from train.train_autoencoder import train_autoencoder
from train.train_resnet import train_resnet
from train.train_densenet import train_densenet
from train.train_hybrid_model import train_hybrid_model

from evaluate.evaluate_models import evaluate_models
# from evaluate.plot_results import plot_detection_metrics # Uncomment if you want to plot from main

from utils.config import config
from utils.logger import setup_logger

def preprocess_data(source_dir, dest_dir, max_images_per_class=500):
    """Preprocess dataset images for training."""
    main_logger = logging.getLogger('MainLogger')
    main_logger.info(f"Starting preprocessing from {source_dir} to {dest_dir}")
    
    # Create processed directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Determine dataset structure (handle different folder layouts)
    if os.path.exists(os.path.join(source_dir, 'real_vs_fake', 'real-vs-fake')):
        dataset_root = os.path.join(source_dir, 'real_vs_fake', 'real-vs-fake')
        real_folder = os.path.join(dataset_root, 'train', 'real')
        fake_folder = os.path.join(dataset_root, 'train', 'fake')
    elif os.path.exists(os.path.join(source_dir, 'real-vs-fake')):
        dataset_root = os.path.join(source_dir, 'real-vs-fake')
        real_folder = os.path.join(dataset_root, 'train', 'real')
        fake_folder = os.path.join(dataset_root, 'train', 'fake')
    elif os.path.exists(os.path.join(source_dir, 'real')):
        # Flatter structure
        real_folder = os.path.join(source_dir, 'real')
        fake_folder = os.path.join(source_dir, 'fake')
    else:
        main_logger.error(f"Could not find dataset structure in {source_dir}")
        return
    
    # Process real images
    real_count = 0
    for img_path in glob.glob(os.path.join(real_folder, '*.jpg')):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, (224, 224))
                out_path = os.path.join(dest_dir, f'real_{real_count:04d}.jpg')
                cv2.imwrite(out_path, resized)
                real_count += 1
                if real_count % 100 == 0:
                    main_logger.info(f"Processed {real_count} real images")
                if real_count >= max_images_per_class:
                    break
        except Exception as e:
            main_logger.error(f"Error processing {img_path}: {e}")
    
    # Process fake images
    fake_count = 0
    for img_path in glob.glob(os.path.join(fake_folder, '*.jpg')):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, (224, 224))
                out_path = os.path.join(dest_dir, f'fake_{fake_count:04d}.jpg')
                cv2.imwrite(out_path, resized)
                fake_count += 1
                if fake_count % 100 == 0:
                    main_logger.info(f"Processed {fake_count} fake images")
                if fake_count >= max_images_per_class:
                    break
        except Exception as e:
            main_logger.error(f"Error processing {img_path}: {e}")
    
    main_logger.info(f"Finished preprocessing: {real_count} real images and {fake_count} fake images")
    return real_count + fake_count

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection and Generation Analysis')
    parser.add_argument('command', type=str, choices=['train_gan', 'train_autoencoder', 'train_resnet', 'train_densenet', 'train_hybrid', 'evaluate', 'generate', 'preprocess'],
                        help='Command to run (train_gan, train_autoencoder, train_resnet, train_densenet, train_hybrid, evaluate, generate, preprocess)')
    parser.add_argument('--model', type=str, choices=['resnet', 'densenet', 'hybrid', 'all'], default='all', help='Model to evaluate (for evaluate command)')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Directory for raw dataset')
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR, help='Directory for processed dataset')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation or resuming training')
    parser.add_argument('--images_per_class', type=int, default=500, help='Number of images per class to use (for preprocess command)')
    # Add more arguments as needed for generate and preprocess commands

    args = parser.parse_args()

    # Update config with command line arguments
    config.update(
        DATA_DIR=args.data_dir,
        PROCESSED_DIR=args.processed_dir,
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        # Update other config settings based on args
    )

    # Setup main logger
    main_log_file = os.path.join(config.LOG_DIR, 'main.log')
    main_logger = setup_logger('MainLogger', main_log_file)
    main_logger.info(f"Command received: {args.command}")

    if args.command == 'train_gan':
        train_gan()
    elif args.command == 'train_autoencoder':
        train_autoencoder()
    elif args.command == 'train_resnet':
        train_resnet()
    elif args.command == 'train_densenet':
        train_densenet()
    elif args.command == 'train_hybrid':
        train_hybrid_model()
    elif args.command == 'evaluate':
        # Evaluation logic will need to handle selecting specific models or all
        if args.model == 'all':
            evaluation_results = evaluate_models() # evaluate_models handles iterating through defined models
        else:
            # Evaluate a specific model using the provided --checkpoint (mandatory for now)
            import torch
            from data.dataloader import create_dataloader
            from evaluate.evaluate_models import load_model, evaluate_model as eval_single_model

            if args.checkpoint is None:
                main_logger.error("--checkpoint must be provided when evaluating a single model.")
                return
            if not os.path.exists(args.checkpoint):
                main_logger.error(f"Checkpoint file not found at {args.checkpoint}")
                return

            device = torch.device(config.DEVICE)
            main_logger.info(f"Evaluating {args.model} using checkpoint {args.checkpoint} on device {device}")

            # Create evaluation dataloader (no shuffle) - use test split
            eval_dataloader = create_dataloader(config.PROCESSED_DIR, config.BATCH_SIZE, shuffle=False, split='test')

            try:
                model = load_model(args.model, args.checkpoint, num_classes=2, device=device)
                metrics = eval_single_model(model, eval_dataloader, device)
                main_logger.info(f"{args.model} Evaluation Metrics: Accuracy: {metrics['accuracy']:.4f}, "
                                 f"F1 Score: {metrics['f1_score']:.4f}")
                main_logger.info(f"{args.model} Confusion Matrix:\n{metrics['confusion_matrix']}")
            except Exception as e:
                main_logger.error(f"Error during evaluation: {e}")
                return

            # Store results in evaluation_results for potential further use
            evaluation_results = {args.model: metrics}
        # You might want to plot results here after evaluation
        # if evaluation_results:
        #     plot_detection_metrics(evaluation_results, save_path=os.path.join(config.RESULT_DIR, 'final_evaluation_metrics.png'))
    elif args.command == 'generate':
        # Implement deepfake generation logic here
        main_logger.info("Generate command is not yet implemented.")
        pass
    elif args.command == 'preprocess':
        # Preprocessing logic - prepare images from dataset to processed folder
        dataset_dir = args.data_dir
        if not os.path.exists(dataset_dir):
            # Try dataset folder in current directory if data_dir doesn't exist
            dataset_dir = os.path.join(os.getcwd(), 'dataset')
            if not os.path.exists(dataset_dir):
                main_logger.error(f"Dataset directory not found at {args.data_dir} or {dataset_dir}")
                return
        
        processed_count = preprocess_data(
            dataset_dir, 
            args.processed_dir, 
            max_images_per_class=args.images_per_class
        )
        
        main_logger.info(f"Preprocessing complete! {processed_count} total images processed.")
        main_logger.info(f"You can now train models using: python main.py train_resnet --processed_dir {args.processed_dir}")
    else:
        main_logger.error("Invalid command.")
        parser.print_help()

if __name__ == '__main__':
    main() 

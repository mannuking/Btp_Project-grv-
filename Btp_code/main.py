import argparse
import os

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
            # Evaluate a specific model - needs implementation in evaluate_models.py
            main_logger.warning("Evaluating a single model is not yet fully implemented in main.py.")
            # You would call evaluate_model with the specific model and checkpoint path
            # metrics = evaluate_model(loaded_model, eval_dataloader, device)
            pass
        # You might want to plot results here after evaluation
        # if evaluation_results:
        #     plot_detection_metrics(evaluation_results, save_path=os.path.join(config.RESULT_DIR, 'final_evaluation_metrics.png'))
    elif args.command == 'generate':
        # Implement deepfake generation logic here
        main_logger.info("Generate command is not yet implemented.")
        pass
    elif args.command == 'preprocess':
        # Implement data preprocessing logic here
        main_logger.info("Preprocess command is not yet implemented.")
        pass
    else:
        main_logger.error("Invalid command.")
        parser.print_help()

if __name__ == '__main__':
    main() 
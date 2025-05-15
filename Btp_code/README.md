# Deepfake Detection and Generation Analysis

## Project Overview

This project focuses on both generating deepfakes using Generative Adversarial Networks (GANs) and Autoencoders, and detecting them using various Convolutional Neural Network (CNN) based models, including ResNet, DenseNet, and a custom Hybrid model.

## Project Structure

```
deepfake_analysis/
│
├── 📁 data/                     # Datasets and dataset utilities
│   ├── celebA/                 # Raw dataset (CelebA or similar) - **Requires manual download**
│   ├── processed/              # Preprocessed/augmented images
│   └── dataloader.py          # Dataset class and transforms
│
├── 📁 models/                  # All deep learning models
│   ├── __init__.py
│   ├── gan.py                 # DCGAN or GAN model
│   ├── autoencoder.py         # Autoencoder and VAE
│   ├── cnn_base.py            # Base CNN for detection
│   ├── resnet.py              # ResNet architecture
│   ├── densenet.py            # DenseNet implementation
│   └── hybrid_model.py        # Custom hybrid model
│
├── 📁 train/                   # Training scripts
│   ├── train_gan.py
│   ├── train_autoencoder.py
│   ├── train_resnet.py
│   ├── train_densenet.py
│   └── train_hybrid.py
│
├── 📁 evaluate/                # Evaluation and results
│   ├── evaluate_models.py     # Accuracy, confusion matrix, Grad-CAM
│   └── plot_results.py        # Plotting training/validation curves
│
├── 📁 utils/                   # Helper functions
│   ├── visualization.py       # Grad-CAM, feature maps, etc.
│   ├── metrics.py             # Accuracy, F1, etc.
│   ├── config.py              # Global settings (batch size, lr, etc.)
│   └── logger.py              # Logging and checkpointing
│
├── 📁 notebooks/              # Jupyter notebooks for EDA and testing - **Empty for now**
│   └── experiment_logs.ipynb
│
├── main.py                    # Entry point: CLI to run generation/detection
├── README.md                    # Project overview and instructions
└── requirements.txt           # Python dependencies
└── .gitignore                 # Files to ignore in version control
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd deepfake_analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download and preprocess the dataset:**
    *   Download the chosen dataset (e.g., CelebA). Place the raw data in the `data/celebA/` directory.
    *   **Note:** The data preprocessing script (`main.py preprocess`) is not yet implemented. You will need to add the logic to load the raw data, perform necessary transformations (resizing, normalization, etc.), and save the processed data (including labels for real/fake) in the `data/processed/` directory. The `data/dataloader.py` file assumes data is available in `data/processed/` and image files determine the labels (which needs refinement based on your actual dataset structure).

## Usage

The `main.py` script provides a command-line interface to run different parts of the project.

```bash
python main.py <command> [options]
```

**Available Commands:**

*   `train_gan`: Train the GAN model.
*   `train_autoencoder`: Train the Autoencoder model.
*   `train_resnet`: Train the ResNet detection model.
*   `train_densenet`: Train the DenseNet detection model.
*   `train_hybrid`: Train the custom Hybrid detection model.
*   `evaluate`: Evaluate the trained detection models.
*   `generate`: (Not yet implemented) Generate deepfake images using a trained generation model.
*   `preprocess`: (Not yet implemented) Preprocess the raw dataset.

**Common Options:**

*   `--epochs <int>`: Number of training epochs (overrides config).
*   `--batch_size <int>`: Batch size (overrides config).
*   `--lr <float>`: Learning rate (overrides config).
*   `--data_dir <path>`: Path to the raw data directory (overrides config).
*   `--processed_dir <path>`: Path to the processed data directory (overrides config).
*   `--checkpoint <path>`: Path to a model checkpoint for resuming training or evaluation.

**Evaluate Command Options:**

*   `--model {resnet,densenet,hybrid,all}`: Specify which model(s) to evaluate (default: `all`).

**Examples:**

*   Train the ResNet model for 50 epochs with a batch size of 64:
    ```bash
    python main.py train_resnet --epochs 50 --batch_size 64
    ```

*   Evaluate all trained detection models:
    ```bash
    python main.py evaluate
    ```

*   Evaluate the DenseNet model using a specific checkpoint:
    ```bash
    python main.py evaluate --model densenet --checkpoint checkpoints/densenet_epoch_X.pth
    ```
    (Remember to replace `X` with the actual epoch number of your checkpoint)

## Next Steps / To Do

*   Implement the `preprocess` command in `main.py` and add data loading/preprocessing logic.
*   Refine `data/dataloader.py` to handle loading processed data and corresponding labels correctly.
*   Implement the `generate` command in `main.py`.
*   Complete the implementation of Grad-CAM visualization in `utils/visualization.py` and integrate it into `evaluate/evaluate_models.py`.
*   Add functionality to `evaluate/plot_results.py` to plot confusion matrices.
*   Add more detailed model architectures (e.g., specific DCGAN implementation, VAE).
*   Implement training and validation loops with proper dataset splitting in the training scripts.
*   Add more advanced evaluation metrics if necessary.
*   Create Jupyter notebooks in `notebooks/` for data exploration and testing small code snippets. 
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(predictions, labels):
    """Calculates accuracy, F1 score, and confusion matrix."""
    # Ensure predictions and labels are on CPU and converted to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, list):
        predictions = np.array(predictions)
        
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # Assuming binary classification (deepfake or real)
    # If using sigmoid output, convert probabilities to class labels (0 or 1)
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # Assuming multi-class output (e.g., softmax), take argmax
        predicted_classes = np.argmax(predictions, axis=1)
    elif np.max(predictions) <= 1 and np.min(predictions) >= 0:
         # Assuming sigmoid output, threshold at 0.5
         predicted_classes = (predictions > 0.5).astype(int)
    else:
        # Assuming predictions are already class labels
        predicted_classes = predictions

    accuracy = accuracy_score(labels, predicted_classes)
    # Use 'binary' for binary classification, 'weighted' or 'macro' for multi-class
    f1 = f1_score(labels, predicted_classes, average='binary')
    cm = confusion_matrix(labels, predicted_classes)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }

# Example Usage
if __name__ == '__main__':
    # Dummy data (example for binary classification)
    true_labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1, 0, 0]) # 0: real, 1: fake
    # Example predictions (e.g., from a model with sigmoid output)
    model_predictions = torch.tensor([0.1, 0.9, 0.3, 0.6, 0.8, 0.4, 0.2, 0.7, 0.0, 0.55])

    metrics = calculate_metrics(model_predictions, true_labels)

    print("Calculated Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print("  Confusion Matrix:")
    print(metrics['confusion_matrix']) 

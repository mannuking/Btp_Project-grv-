import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet, self).__init__()

        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1) # Use DenseNet121 with ImageNet weights
            # Replace the last fully connected layer
            num_ftrs = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            # Basic DenseNet-like structure without pre-training (simplified)
            self.densenet = models.densenet121(weights=None) # Use DenseNet121 architecture
            num_ftrs = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Instantiate DenseNet model
    # Use pretrained=True for transfer learning, False for training from scratch
    densenet_model = DenseNet(num_classes=2, pretrained=True)

    # Create a dummy image batch (assuming same shape as expected by DenseNet input, typically 3 channels, 224x224 or similar)
    # Note: If your data is 128x128 as per config, you might need to adjust transforms or use a different model.
    # For this example, let's assume input is resized to 224x224 for the standard DenseNet
    dummy_input = torch.randn(config.BATCH_SIZE, 3, 224, 224) # Example input shape for DenseNet121

    # Pass dummy input through the model
    output = densenet_model(dummy_input)
    print(f"Shape of DenseNet output: {output.shape}") 
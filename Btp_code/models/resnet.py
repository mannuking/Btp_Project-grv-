import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet, self).__init__()

        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Use ResNet50 with ImageNet weights
            # Replace the last fully connected layer
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        else:
            # Basic ResNet-like structure without pre-training (simplified)
            self.resnet = models.resnet50(weights=None) # Use ResNet50 architecture
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Instantiate ResNet model
    # Use pretrained=True for transfer learning, False for training from scratch
    resnet_model = ResNet(num_classes=2, pretrained=True)

    # Create a dummy image batch (assuming same shape as expected by ResNet input, typically 3 channels, 224x224 or similar)
    # Note: If your data is 128x128 as per config, you might need to adjust transforms or use a different model.
    # For this example, let's assume input is resized to 224x224 for the standard ResNet
    dummy_input = torch.randn(config.BATCH_SIZE, 3, 224, 224) # Example input shape for ResNet50

    # Pass dummy input through the model
    output = resnet_model(dummy_input)
    print(f"Shape of ResNet output: {output.shape}") 
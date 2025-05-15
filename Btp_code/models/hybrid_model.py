import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super(HybridModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Add more layers or integrate other model components here
        )

        # Determine the size of the flattened features after convolutional layers
        # This depends on the input image size and the pooling layers
        # A common way to handle this is to pass a dummy tensor through the feature extractor
        # Or make the classifier adaptive (e.g., using AdaptiveAvgPool2d)

        # For simplicity, let's use AdaptiveAvgPool2d and a linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes) # 128 is the number of channels after the last conv layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# Example Usage
if __name__ == '__main__':
    from utils.config import config

    # Instantiate HybridModel
    hybrid_model = HybridModel(num_classes=2, input_channels=config.CHANNELS)

    # Create a dummy image batch (using config dimensions)
    dummy_input = torch.randn(config.BATCH_SIZE, config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)

    # Pass dummy input through the model
    output = hybrid_model(dummy_input)
    print(f"Shape of HybridModel output: {output.shape}") 
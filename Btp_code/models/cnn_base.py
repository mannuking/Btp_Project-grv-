import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseCNN, self).__init__()
        # Base convolutional layers (to be implemented by subclasses)
        self.features = nn.Sequential()
        # Classifier (to be implemented by subclasses or in this base class)
        self.classifier = nn.Identity() # Placeholder

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

# Example of how a subclass might use it
# class SpecificCNN(BaseCNN):
#     def __init__(self, num_classes=2):
#         super(SpecificCNN, self).__init__(num_classes=num_classes)
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#             # Add more layers...
#         )
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(32, num_classes)
#         )

# Example usage (assuming you have input tensor 'dummy_input')
# if __name__ == '__main__':
#     dummy_input = torch.randn(1, 3, 128, 128) # Example batch of 1, 3 channels, 128x128 image
#     model = BaseCNN(num_classes=2)
#     # Note: BaseCNN by itself won't do much as features and classifier are empty/identity
#     # print(model(dummy_input).shape) # This would likely fail or give unexpected results
#
#     # You would typically instantiate a subclass like SpecificCNN
#     # specific_model = SpecificCNN(num_classes=2)
#     # print(specific_model(dummy_input).shape) # This should work if SpecificCNN is implemented 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLatentDiffusionModel(nn.Module):
    def __init__(self, feature_channels=1, target_height=128, target_width=128):
        super(SimpleLatentDiffusionModel, self).__init__()
        # We will create a model that simply upscales the input.
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 32 -> 64
        self.conv1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64 -> 128
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        x = self.upsample2(x)
        x = F.relu(self.conv2(x))
        return x

# Assuming GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the appropriate device
model = SimpleLatentDiffusionModel(feature_channels=1, target_height=128, target_width=128).to(device)

# Example latent space input (assuming 3 channels as planes but each with 1 feature)
latent_space_input = torch.randn(1, 3, 1, 32, 32).to(device)  # (batch_size, planes, features, height, width)

# Reshape to (batch_size * planes, features, height, width) for the model
latent_space_input = latent_space_input.view(-1, 1, 32, 32)  # This becomes (3, 1, 32, 32)

# Forward pass through the model
output = model(latent_space_input)

# Reshape output back to (batch_size, planes, features, height, width)
output = output.view(1, 3, 1, 128, 128)  # Should now be (1, 3, 1, 128, 128)

print("Output shape:", output.shape)  # Expected: torch.Size([1, 3, 1, 128, 128])

import torch
import torch.nn as nn

class UNetWithCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_feature_channels = 3
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.num_feature_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Cross attention layer
        self.cross_attention = CrossAttention(256, 1536)
        
        # Decoder with upsampling
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Note concatenation increases input channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Conv2d(64, self.num_feature_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x, embedding):
        # Encoding with skip connections
        x1 = self.enc1(x)   # Encoder block 1 output (64 channels)
        x2 = self.enc2(x1)  # Encoder block 2 output (128 channels)
        x3 = self.enc3(x2)  # Encoder block 3 output (256 channels)
        
        # Apply cross-attention on the deepest encoded layer (x3)
        x3 = self.cross_attention(x3, embedding)

        # Decoding with skip connections
        x = self.dec1(torch.cat([x3, x2], dim=1))  # Concatenate encoder output x2 with decoder output x3
        x = self.dec2(torch.cat([x, x1], dim=1))   # Concatenate encoder output x1 with decoder output x
        x = self.dec3(x)  # No concatenation needed for the final layer
        x = self.final_activation(x)
        return x
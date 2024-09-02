import torch
from torch import nn, optim
import torch.nn.functional as F
from openai import OpenAI
import os
import numpy as np
from data_loader import latent_dataloader, embedding_dataloader, triplane_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_conv = nn.Linear(embed_dim, feature_dim)
        self.value_conv = nn.Linear(embed_dim, feature_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x, embedding):
        # x: feature maps from the UNet [batch_size, feature_dim, H, W]
        # embedding: text embeddings [batch_size, embed_dim]

        batch_size, feature_dim, H, W = x.shape

        # Prepare queries from feature maps
        query = self.query_conv(x)
        query = query.view(batch_size, feature_dim, -1).permute(2, 0, 1)  # [HW, batch_size, feature_dim]

        # Prepare keys and values from embeddings
        key = self.key_conv(embedding)
        value = self.value_conv(embedding)
        key = key.unsqueeze(0).repeat(H * W, 1, 1)  # Repeat keys for each spatial location
        value = value.unsqueeze(0).repeat(H * W, 1, 1)  # Repeat values for each spatial location

        # Compute attention
        attended, _ = self.attention(query, key, value)
        attended = attended.permute(1, 2, 0).view(batch_size, feature_dim, H, W)

        # Combine attended features and input features
        combined_features = x + attended
        return combined_features

class UNetWithCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # num_feature_channels = 3 * 4
        num_feature_channels = 3 * 3
        # Define the standard UNet layers
        self.enc1 = nn.Conv2d(num_feature_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Cross attention layer
        self.cross_attention = CrossAttention(256, 1536)
        
        # Continue with the rest of the UNet
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Conv2d(64, num_feature_channels, kernel_size=3, padding=1)

    def forward(self, x, embedding):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        
        # Apply cross-attention
        x3 = self.cross_attention(x3, embedding)
        
        # Decoding
        x = F.relu(self.dec1(x3))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x

class CosineNoiseScheduler:
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.betas = np.cos(np.linspace(0, np.pi / 2, num_steps)) ** 2

    def get_noise_factor(self, step):
        return self.betas[step]

def train():
    # Initialize model
    scheduler = CosineNoiseScheduler(1000)
    model = UNetWithCrossAttention().to(device)
    noise_scheduler = CosineNoiseScheduler(1000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    checkpoint_dir = './ldm_checkpoints'
    num_epochs = 100
    num_timesteps = 1000
    model.train()
    latent_data, embedding_data = triplane_dataloader(), embedding_dataloader()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for latent_tensors, embeddings in zip(latent_data, embedding_data):
            b, p, c, h, w = latent_tensors.size()
            latent_tensors = latent_tensors.reshape(b, p * c, h, w).to(device)
            embeddings = embeddings.to(device)
            for step in range(num_timesteps):
                noise_level = scheduler.get_noise_factor(step)
                # Apply noise
                noise = torch.randn_like(latent_tensors) * noise_level
                noisy_latents = latent_tensors + noise
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                reconstructed_noise = model(noisy_latents, embeddings)
                # Compute loss: Predict the noise that was added
                loss = criterion(reconstructed_noise, noise)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / (len(latent_data) * num_timesteps)}")
        if epoch % 10 == 0:
            checkpoint_path = f'{checkpoint_dir}/ldm_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

def main():
    # # Run training
    # train()
    print('Main function: LDM')

if __name__ == '__main__':
    main()
import torch
from torch import nn, optim
import torch.nn.functional as F
from openai import OpenAI
import os
import numpy as np
from data_loader import latent_dataloader, embedding_dataloader, triplane_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, embed_dim, num_heads=1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.query_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
#         self.key_conv = nn.Linear(embed_dim, feature_dim)
#         self.value_conv = nn.Linear(embed_dim, feature_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

#     def forward(self, x, embedding):
#         batch_size, feature_dim, H, W = x.shape
#         query = self.query_conv(x).view(batch_size, feature_dim, -1).permute(2, 0, 1)  # [HW, batch_size, feature_dim]
#         # query = self.query_conv(x)
#         key = self.key_conv(embedding)
#         value = self.value_conv(embedding)
        
#         # Reshape key and value for each head and spatial location
#         key = key.repeat(1, H * W).view(-1, batch_size, feature_dim).permute(1, 0, 2)
#         value = value.repeat(1, H * W).view(-1, batch_size, feature_dim).permute(1, 0, 2)
#         # key = key.unsqueeze(1).expand(-1, H * W, -1).reshape(H * W, batch_size, feature_dim)
#         # value = value.unsqueeze(1).expand(-1, H * W, -1).reshape(H * W, batch_size, feature_dim)

#         attended, _ = self.attention(query, key, value)
#         attended = attended.permute(1, 2, 0).view(batch_size, feature_dim, H, W)
        
#         combined_features = x + attended
#         return combined_features

# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=8):
#         super(CrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.query_dim = 64  # Dimension of the projected query feature maps
        
#         # Linear transformations for query, key, and value
#         self.query_proj = nn.Linear(embed_dim, self.query_dim * num_heads)
#         self.key_proj = nn.Linear(embed_dim, self.query_dim * num_heads)
#         self.value_proj = nn.Linear(embed_dim, self.query_dim * num_heads)

#         # Output projection layer
#         self.output_proj = nn.Linear(self.query_dim * num_heads, 256)

#         # Scaling factor to normalize the dot products
#         self.scale = 1. / (self.query_dim ** 0.5)

#     def forward(self, x, embedding):
#         batch_size, _, height, width = x.shape
#         # Flatten the spatial dimensions
#         x = x.view(batch_size, 256, -1).permute(0, 2, 1)  # Shape: [batch, height*width, channels]
        
#         # Expand the embedding to match the batch size
#         embedding = embedding.repeat(batch_size, 1)

#         # Project queries, keys, values
#         q = self.query_proj(embedding).view(batch_size, self.num_heads, self.query_dim).permute(1, 0, 2)
#         k = self.key_proj(embedding).view(batch_size, self.num_heads, self.query_dim).permute(1, 0, 2)
#         v = self.value_proj(embedding).view(batch_size, self.num_heads, self.query_dim).permute(1, 0, 2)
        
#         # Attention mechanism
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#         attn = F.softmax(attn_scores, dim=-1)
#         context = torch.matmul(attn, v).permute(1, 0, 2).contiguous()
        
#         # Concatenate heads and project output
#         context = context.view(batch_size, -1)
#         output = self.output_proj(context)

#         return output

class CrossAttention(nn.Module):
    def __init__(self, features_dim, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.scale = (self.features_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(features_dim, features_dim, bias=False)
        self.to_kv = nn.Linear(embed_dim, features_dim * 2, bias=False)
        # self.to_out = nn.Linear(features_dim, features_dim)

    def forward(self, x, embedding):
        b, _, h, w = x.shape

        # Query from feature maps
        q = self.to_q(x.flatten(2).transpose(1, 2))  # Shape: (batch_size, height*width, features_dim)
        q = q.view(b, h * w, self.num_heads, self.features_dim // self.num_heads).permute(0, 2, 1, 3)

        # Key and value from embedding vector
        kv = self.to_kv(embedding.expand(b, -1)).view(b, 2, self.num_heads, self.features_dim // self.num_heads).permute(1, 2, 0, 3)
        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)

        # Aggregate values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, self.features_dim)
        out = out.view(b, self.features_dim, h, w)  # Reshape to (batch_size, features_dim, height, width)
        # Final linear transformation
        # out = self.to_out(out)

        return out

class UNetWithCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # num_feature_channels = 3 * 3
        num_feature_channels = 3
        # Define the standard UNet layers with batch normalization
        self.enc1 = nn.Sequential(
            nn.Conv2d(num_feature_channels, 64, kernel_size=3, padding=1),
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
        # Continue with the rest of the UNet with batch normalization
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Conv2d(64, num_feature_channels, kernel_size=3, padding=1)
        # Consider activation if output needs to be bounded (e.g., tanh for [-1, 1])
        self.final_activation = nn.Tanh()  # Uncomment if needed

    def forward(self, x, embedding):
        # Encoding with skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        # Apply cross-attention
        x3 = self.cross_attention(x3, embedding)
        # Decoding with skip connections
        x = self.dec1(x3)
        x = self.dec2(x)
        x = self.dec3(x)
        # Final activation (if needed)
        # x = self.final_activation(x)  # Uncomment if output activation is necessary
        return x


# class CosineNoiseScheduler:
#     def __init__(self, num_steps):
#         self.num_steps = num_steps
#         self.betas = np.cos(np.linspace(0, np.pi / 2, num_steps)) ** 2

#     def get_noise_factor(self, step):
#         return self.betas[step]

class NoiseScheduler:
    def __init__(self, timesteps, schedule_fn):
        self.timesteps = timesteps
        self.betas = schedule_fn(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod).to(device)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        alpha_t = self.sqrt_alpha_cumprod[t]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t]
        one_minus_alpha_t = one_minus_alpha_t.view(-1, 1, 1, 1)
        a, b = alpha_t * x_start + one_minus_alpha_t * noise, noise
        return alpha_t * x_start + one_minus_alpha_t * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas[t] * x_t - self.sqrt_recip_alphas[t] * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t] * noise
        )

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def train():
    # Initialize model
    model = UNetWithCrossAttention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    checkpoint_dir = './ldm_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # num_epochs = 100
    num_epochs = 10
    num_timesteps = 1000
    scheduler = NoiseScheduler(num_timesteps, linear_beta_schedule)
    model.train()
    latent_data = triplane_dataloader()
    embedding_data = embedding_dataloader()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for latent_tensors, embeddings in zip(latent_data, embedding_data):
            b, p, c, h, w = latent_tensors.size()
            latent_tensors = latent_tensors.reshape(b*p, c, h, w).to(device)
            embeddings = embeddings.to(device)
            for step in range(num_timesteps):
                timesteps = torch.randint(0, scheduler.timesteps, (b*p, ), device=device)
                noisy_data, noise = scheduler.add_noise(latent_tensors, timesteps)  # Adding noise
                # noisy_latents = latent_tensors + noisy_data
                optimizer.zero_grad()   # Zero the parameter gradients
                # Forward pass
                reconstructed_noise = model(noisy_data, embeddings)
                loss = criterion(reconstructed_noise, noise)    # Predicting the added noise
                # Backward pass
                loss.backward()
                optimizer.step()    # Optimization
                epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(latent_data)}")
        if epoch % 8 == 0:
            checkpoint_path = f'{checkpoint_dir}/ldm_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

def main():
    # Run training
    print('Main function: LDM')
    train()

if __name__ == '__main__':
    main()
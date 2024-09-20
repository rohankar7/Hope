import torch
import torch.nn as nn
import math
from data_loader import latent_dataloader, embedding_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return alpha_t * x_start + one_minus_alpha_t * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas[t] * x_t - self.sqrt_recip_alphas[t] * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t] * noise
        )

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class CrossAttention(nn.Module):
    def __init__(self, query_dim, embedding_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embedding_dim = embedding_dim
        
        self.query_projection = nn.Linear(query_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        # self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, query_dim)
        self.out_projection = nn.Linear(embedding_dim, query_dim)
        
    def forward(self, x, text_embedding, time_embedding):
        batch_size, channels, height, width = x.size()
        num_features = height * width
        # Combine text and time embeddings
        conditioning_embedding = text_embedding + time_embedding
        # Reshape the feature map (query)
        query = x.view(batch_size, channels, num_features).transpose(1, 2)  # (batch_size, num_features, channels)
        query = self.query_projection(query)  # Project query to embedding dimension
        # Expand conditioning embeddings to match the number of features
        key = self.key_projection(conditioning_embedding).unsqueeze(1)  # (batch_size, 1, embedding_dim)
        value = self.value_projection(conditioning_embedding).unsqueeze(1)  # (batch_size, 1, embedding_dim)
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_features, 1)
        attention_scores = attention_scores / (self.embedding_dim ** 0.5)  # Scale by sqrt of embedding dimension
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_features, 1)
        # Compute the weighted sum of values
        context = torch.matmul(attention_weights, value)  # (batch_size, num_features, embedding_dim)
        # Project back to the original query dimension and reshape
        context = self.out_projection(context).transpose(1, 2).view(batch_size, channels, height, width)
        print((x+context).shape)
        return x + context  # Residual connection

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.cross_attention1 = CrossAttention(out_channels, embedding_dim)
        self.cross_attention2 = CrossAttention(out_channels, embedding_dim)

    def forward(self, x, text_embedding, time_embedding):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cross_attention1(x, text_embedding, time_embedding)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.cross_attention2(x, text_embedding, time_embedding)
        
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, embedding_dim)
        )

    def forward(self, x, text_embedding, time_embedding):
        return self.down(x, text_embedding, time_embedding)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, embedding_dim)

    def forward(self, x1, x2, text_embedding, time_embedding):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, text_embedding, time_embedding)

class UNet(nn.Module):
    def __init__(self, embedding_dim=1536, time_embedding_dim=1536):
        super(UNet, self).__init__()
        self.time_embedding = SinusoidalPositionEmbedding(time_embedding_dim)
        
        # Encoder path
        self.inc = DoubleConv(4, 64, embedding_dim + time_embedding_dim)
        self.down1 = DownBlock(64, 128, embedding_dim + time_embedding_dim)
        self.down2 = DownBlock(128, 256, embedding_dim + time_embedding_dim)
        self.down3 = DownBlock(256, 512, embedding_dim + time_embedding_dim)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, embedding_dim + time_embedding_dim)
        
        # Decoder path
        self.up1 = UpBlock(1024, 512, embedding_dim + time_embedding_dim)
        self.up2 = UpBlock(512, 256, embedding_dim + time_embedding_dim)
        self.up3 = UpBlock(256, 128, embedding_dim + time_embedding_dim)
        self.up4 = UpBlock(128, 64, embedding_dim + time_embedding_dim)
        
        # Final convolution
        self.outc = nn.Conv2d(64, 4, kernel_size=1)
        
    def forward(self, x, text_embedding, timesteps):
        # Compute time embeddings
        time_embedding = self.time_embedding(timesteps)
        
        # Encoder
        x1 = self.inc(x, text_embedding, time_embedding)
        x2 = self.down1(x1, text_embedding, time_embedding)
        x3 = self.down2(x2, text_embedding, time_embedding)
        x4 = self.down3(x3, text_embedding, time_embedding)
        
        # Bottleneck
        x5 = self.bottleneck(x4, text_embedding, time_embedding)
        
        # Decoder
        x = self.up1(x5, x4, text_embedding, time_embedding)
        x = self.up2(x, x3, text_embedding, time_embedding)
        x = self.up3(x, x2, text_embedding, time_embedding)
        x = self.up4(x, x1, text_embedding, time_embedding)
        
        # Final output
        x = self.outc(x)
        return x

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Assuming UNet and other necessary modules are defined as above

# Example training loop
def train():
    model = UNet()
    model.to(device)
    num_epochs = 100
    timesteps = 1000
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()
    scheduler = NoiseScheduler(timesteps, linear_beta_schedule)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
# Assume `encoder` is a pre-trained model that encodes images into latent space
# `decoder` is a model that decodes latent representations back to the original data space

        for i, (latent_target, text_embeddings) in enumerate(zip(latent_dataloader(), embedding_dataloader())):
            # Move data to GPU if available
            batch_size, planes, features, height, width = latent_target.size()
            latent_target = latent_target.view(batch_size * planes, features, height, width).to(device)
            latent_target = latent_target.to(device)
            text_embeddings = text_embeddings.to(device)
                        
            # Add noise to the latent representation for the forward diffusion process
            # noise = torch.randn_like(latent_target)
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size * planes,), device=device)
            # noisy_latent_input = apply_noise(latent_target, noise, timesteps)  # This function applies noise according to the diffusion process
            noisy_data, noise = scheduler.add_noise(latent_target, timesteps)
            # Forward pass
            noisy_data.to(device)
            optimizer.zero_grad()
            predicted_latent = model(noisy_data, text_embeddings, timesteps)
            loss = loss_function(predicted_latent, latent_target) 
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

# Example usage
# Assuming we have a dataloader that yields (latent_input, text_embeddings, target)
# latent_input shape: (batch_size, 3, 4, 32, 32)
# text_embeddings shape: (batch_size, 1536)
# target shape: (batch_size, 3, 4, 32, 32)
# Train the model
train()
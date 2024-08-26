import torch
from torch import nn, optim
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_loader import latent_dataloader
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

timesteps = 1000
scheduler = NoiseScheduler(timesteps, linear_beta_schedule)

class SimpleLatentDiffusionModel(nn.Module):
    def __init__(self, feature_channels=4, target_height=128, target_width=128):
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


# output = model(latent_space_input)
# output = output.view(1, 3, 1, 128, 128)  # Should now be (1, 3, 1, 128, 128)
# print("Output shape:", output.shape)  # Expected: torch.Size([1, 3, 1, 128, 128])

def train_simple_latent_diffusion_model(scheduler, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 32
    model = SimpleLatentDiffusionModel(feature_channels=4, target_height=128, target_width=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    latent_data = latent_dataloader()
    num_epochs = 100
    timesteps = 1000
    scheduler = NoiseScheduler(timesteps, linear_beta_schedule)
    checkpoint_dir = './ldm_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in latent_data:
            batch_data = batch  # Directly use the batch tensor
            batch_size, num_planes, features, height, width = batch_data.size()
            batch_data = batch_data.view(batch_size * num_planes, features, height, width).to(device)
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size * num_planes,), device=device)

            noisy_data, noise = scheduler.add_noise(batch_data, timesteps)

            optimizer.zero_grad()
            outputs = model(noisy_data)  # Pass only the noisy_data to the model

            # Upsample noise to match the output size
            target = F.interpolate(noise, size=(128, 128), mode='bilinear', align_corners=False)
            
            loss = F.mse_loss(outputs, target)  # Loss between predicted noise and actual noise
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(latent_data)}')
        if epoch % 10 == 0:
            checkpoint_path = f'{checkpoint_dir}/ldm_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Example usage

train_simple_latent_diffusion_model(scheduler)
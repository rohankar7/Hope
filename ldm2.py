import torch
from torch import nn, optim
import torch.nn.functional as F
import os
from data_loader import latent_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.pool(x1)
        return x2, x1

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LatentDiffusionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(LatentDiffusionModel, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024 + 512, 512)  # Adjust input channels after concatenation
        self.up2 = UpBlock(512 + 256, 256)  # Adjust input channels after concatenation
        self.up3 = UpBlock(256 + 128, 128)  # Adjust input channels after concatenation
        self.up4 = UpBlock(128 + 64, 64)    # Adjust input channels after concatenation
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2, x1_out = self.down1(x1)
        x3, x2_out = self.down2(x2)
        x4, x3_out = self.down3(x3)
        x5, x4_out = self.down4(x4)
        x = self.up1(x5, x4_out)
        x = self.up2(x, x3_out)
        x = self.up3(x, x2_out)
        x = self.up4(x, x1_out)
        return self.outc(x)

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

def train_latent_diffusion_model():
    latent_dim = 32
    model = LatentDiffusionModel(in_channels=4, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    latent_data = latent_dataloader()
    checkpoint_dir = './ldm_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    timesteps = 1000
    scheduler = NoiseScheduler(timesteps, linear_beta_schedule)
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in latent_data:
            batch_data = batch  # Directly use the batch tensor
            batch_size, num_planes, features, height, width = batch_data.size()
            batch_data = batch_data.view(batch_size * num_planes, features, height, width).to(device)
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size * num_planes,), device=device)
            cond = torch.randn(batch_size * num_planes, 32, device=device)  # Example condition

            noisy_data, noise = scheduler.add_noise(batch_data, timesteps)

            optimizer.zero_grad()
            outputs = model(noisy_data) # Change
            # Upsample noise to match the output size
            # noise = F.interpolate(noise, size=(128, 128), mode='bilinear', align_corners=False)
            loss = F.mse_loss(outputs, noise)  # Loss between predicted noise and actual noise
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

train_latent_diffusion_model()
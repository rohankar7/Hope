import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import latent_dataloader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO
def some_refinement_function(model):
    return

# TODO
def refine_model(coarse_model):
    # Implement refinement process using SDS or other techniques
    # This could be by using iterative optimization to improve the model
    refined_model = some_refinement_function(coarse_model)
    return refined_model

# TODO
# refined_model = refine_model(coarse_model)

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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels=0, multiplier=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(multiplier * in_channels + cond_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        super().__init__()
        self.down = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, cond_channels=cond_channels)
        )
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, c):
        _, _, w, h = x.size()
        c = c.expand(-1, -1, w, h)  # Shape conditional input to match image
        x = self.down(torch.cat([x,c], 1)) # Convolutions over image + condition
        x_small = self.pooling(x)   # Downsample output for next block
        return x, x_small

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear: self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else: self.upsample = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2) # Ask why?
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.Upsample(scale_factor=2)
        self.up = nn.Sequential(
            DoubleConv(in_channels, out_channels, multiplier=2)
        )

    def forward(self, x_small, x_big):
        x_upsampled = self.upsample(x_small)
        # input is CHW
        diffY = x_big.size()[2] - x_upsampled.size()[2]
        diffX = x_big.size()[3] - x_upsampled.size()[3]

        x_upsampled = F.pad(x_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]) # concatenate along the channels axis
        x = torch.cat([x_big, x_upsampled], dim=1)
        return self.up(x)

class LatentDiffusionModel(nn.Module):
    def __init__(self, in_channels, cond_channels, time_embed_dim, latent_dim, out_channels, steps=1000):
        super().__init__()
        self.time_embedding = nn.Embedding(steps, time_embed_dim)
        self.cond_projection = nn.Linear(cond_channels, latent_dim)
        self.in_channels = DoubleConv(in_channels, 64)
        self.out_channels = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128, latent_dim)
        self.down2 = DownBlock(128, 256, latent_dim)
        # self.down3 = DownBlock(256, 512, latent_dim)
        # self.down4 = DownBlock(512, 512, latent_dim)
        self.bottleneck = DoubleConv(256 + latent_dim, 512)
        # self.up1 = UpBlock(1024, 256)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t, cond):
        b, c, h, w = x.shape
        print(x.shape, t.shape, cond.shape)
        timestep = self.time_embedding(t)
        timestep = timestep.view(timestep.size(0), timestep.size(1)).unsqueeze(2).unsqueeze(3)
        # print(timestep.shape)
        cond = self.cond_projection(cond)
        cond = cond.view(cond.size(0), cond.size(1)).unsqueeze(2).unsqueeze(3)
        # print(cond.shape)
        timestep = self.time_embedding(t).view(b, -1, 1, 1)
        # print('Timestep', timestep.shape)
        timestep = timestep.expand(b, timestep.size(1), h, w)  # Expand timestep to match cond's spatial dimensions
        cond = self.cond_projection(cond).view(b, -1, 1, 1)
        cond = cond.expand(b, cond.size(1), h, w)  # Expand cond to match spatial dimensions
        cond = torch.cat([timestep, cond], dim=1)
        # cond = torch.cat([timestep, cond], dim=1)
        # cond = torch.cat([timestep, cond], dim=1)
        x1 = self.inc(x)
        x2, x1_small = self.down1(x1, cond)
        x3, x2_small = self.down2(x2, cond)
        # x4, x3_small = self.down3(x3, cond)
        x_bottleneck = self.bottleneck(torch.cat([x3, cond.expand_as(x3)], dim=1))
        x = self.up1(x2_small, x_bottleneck)
        x = self.up2(x1_small, x)
        x = self.up3(x1, x)
        logits = self.outc(x)
        return logits

def train_ldm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 32
    ldm = LatentDiffusionModel(in_channels=4, cond_channels=32, time_embed_dim=32, latent_dim=latent_dim, out_channels=4).to(device)
    optimizer = optim.Adam(ldm.parameters(), lr=1e-4)
    ldm.train()
    latent_data = latent_dataloader()
    num_epochs = 100
    timesteps = 1000
    scheduler = NoiseScheduler(timesteps, linear_beta_schedule)
    checkpoint_dir = './ldm_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Training loop
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        for latent in latent_data:
            batch_size, planes, features, height, width = latent.size()
            latent = latent.view(batch_size * planes, features, height, width).to(device)
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size * planes,), device=device)
            cond = torch.randn(batch_size * planes, 32, device=device)
            noisy_data, noise = scheduler.add_noise(latent, timesteps)
            optimizer.zero_grad()
            outputs = ldm(noisy_data, timesteps, cond)
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
                'model_state_dict': ldm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def main():
    # Training the ldm
    train_ldm() # Uncomment during training

if __name__ == '__main__':
    main()
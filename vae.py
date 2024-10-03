import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import triplane_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from triplane import viz_projections
import config

# Weights:
# 7: Epoch 200, Loss: 0.0016335381509270518

# TODO: Check if the triplane torch images are displaying correctly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
latent_dimension = 16
latent_channels_dim = 16
num_channels = 3
hidden_dim_1 = 10
hidden_dim_2 = 12
weights_dir = 'weights_'
num_planes= 3
# latent_shape = 3 * 3 * 32 * 32

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            # nn.Conv2d(num_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(num_planes * num_channels, hidden_dim_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim_1),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            # nn.Dropout(0.2),
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim_2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(hidden_dim_2, latent_channels_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.flattened_dim = latent_channels_dim * latent_dimension * latent_dimension
        self.hidden_dim = num_planes * num_channels * latent_dimension * latent_dimension
        self.fc = nn.Linear(self.flattened_dim, self.hidden_dim)
        self.dc = nn.Linear(self.hidden_dim, self.flattened_dim)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels_dim, hidden_dim_2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim_2),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(hidden_dim_2, hidden_dim_1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim_1),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(hidden_dim_1, num_planes * num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flattened_dim)
        return self.fc(x), self.fc(x) # For mu and logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        # return eps.mul(std).add_(mu)
        # If not training then return mu
    
    def decode(self, z):
        z = self.dc(z)
        # z = z.reshape(3, num_channels, latent_dimension, latent_dimension)
        z = z.view(-1, latent_channels_dim, latent_dimension, latent_dimension)
        z = self.decoder_conv(z)
        z = z.view(num_planes, num_channels, config.triplane_resolution, config.triplane_resolution)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        mu, logvar = mu.squeeze(), logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def tv_loss(x):
    batch_size, channels, h_x, w_x = x.size()
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    tv_h = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    tv_w = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    # return ((tv_h / count_h) + (tv_w / count_w)) / (batch_size * channels)
    return (tv_h + tv_w) / (batch_size * channels * h_x * w_x)

def vae_loss(recon_x, x, mu, logvar, epoch, num_epochs):
    # print(recon_x.shape, x.shape)
    mse = F.mse_loss(recon_x, x.squeeze(), reduction='mean')
    # mse = F.binary_cross_entropy(recon_x, x, reduction='mean')
    # kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    beta = min(1.0, epoch / (num_epochs * 0.3))  # Increase beta over the first 30% of epochs
    # kld = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    lambda_tv=1e-3
    tvl = tv_loss(recon_x)
    return mse + beta * kld + (lambda_tv * tvl)

def train_vae():
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE().to(device)
    # optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = optim.Adam(vae.parameters(), lr=5e-4)
    # optimizer = optim.Adam(vae.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.1, patience=5, cooldown=5)
    num_epochs = 500
    best_loss = torch.inf
    for epoch in range(num_epochs):
        epoch_loss = 0
        for triplanes in tqdm(triplane_dataloader(), desc=f'Epoch {epoch + 1} / {num_epochs} - Training'):
        # for triplanes in triplane_dataloader():
            optimizer.zero_grad()
            triplanes = triplanes.to(device)
            b, p, c, h, w = triplanes.size()
            # triplanes = triplanes.view(b, p * c, h, w)
            recon_triplane, mu, logvar = vae(triplanes.view(b, p * c, h, w))
            loss = vae_loss(recon_triplane, triplanes, mu, logvar, epoch, num_epochs)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        loss_avg = epoch_loss / len(triplane_dataloader())
        scheduler.step(loss_avg)
        print(f'Epoch {epoch+1}, Loss: {loss_avg}')
        if best_loss > loss_avg:
            best_loss = loss_avg
            torch.save(vae.state_dict(), f'{save_path}/{weights_dir}.pth')

def save_latent_representation():
    vae  = VAE().to(device)
    vae_weights_dir = f'./vae_weights/{weights_dir}.pth'
    vae.load_state_dict(torch.load(vae_weights_dir))
    vae.eval()  # Set the VAE to evaluation mode
    latent_output_dir = './latent_images'
    os.makedirs(latent_output_dir, exist_ok=True)
    with torch.no_grad():
        for i, triplanes in enumerate(triplane_dataloader()):
            triplanes = triplanes.to(device)
            b, p, c, h, w = triplanes.size()
            triplanes = triplanes.view(b, p * c, h, w)
            mu, logvar = vae.encode(triplanes)
            latent_representation = torch.cat([mu, logvar], dim=1)
            latent_path = os.path.join(latent_output_dir, f'latent_{i+1}.pt')
            torch.save(latent_representation.cpu(), latent_path)    # latent shape: batch_size * num_planes(3) x num_features(3) x height(32) x width(32)
            z_reparametrized = vae.reparameterize(mu, logvar)
            z_decoded = vae.decode(z_reparametrized)
            z_decoded = z_decoded.cpu().permute(0, 2, 3, 1).contiguous().numpy()
            viz_projections(z_decoded[0], z_decoded[1], z_decoded[2])

def main():
    print('Main function: VAE')
    train_vae()
    save_latent_representation()

if __name__ == '__main__':
    main()
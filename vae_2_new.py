import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import triplane_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
from triplane import viz_projections
import config
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
latent_dimension = 32
num_channels = 3
hidden_dim_1 = 6
# hidden_dim_2 = 12
latent_channels_dim = 12
weights_dir = 'weights_aeroplanes'
num_planes= 3

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            # nn.Conv2d(num_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(num_channels, hidden_dim_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim_1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            # nn.Dropout(0.1),
            nn.Conv2d(hidden_dim_1, latent_channels_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels_dim),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.flattened_dim = latent_channels_dim * latent_dimension * latent_dimension
        self.hidden_dim = num_channels * latent_dimension * latent_dimension
        self.fc_mu = nn.Linear(self.flattened_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, self.hidden_dim)
        self.dc = nn.Linear(self.hidden_dim, self.flattened_dim)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels_dim, hidden_dim_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim_1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(hidden_dim_1, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flattened_dim)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        # return eps.mul(std).add_(mu)
        # If not training then return mu
    
    def decode(self, z):
        z = self.dc(z)
        z = z.view(-1, latent_channels_dim, latent_dimension, latent_dimension)
        return self.decoder_conv(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def tv_loss(x):
    batch_size, channels, h_x, w_x = x.size()
    tv_h = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    tv_w = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    return (tv_h + tv_w) / (batch_size * channels * h_x * w_x)

def vae_loss(recon_x, x, mu, logvar, epoch, num_epochs):
    mse = F.mse_loss(recon_x, x, reduction='mean')
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # beta = min(1.0, epoch / (num_epochs * 0.3))  # Increase beta over the first 30% of epochs
    beta = 1e-5 * 0
    # beta = 1 * 0
    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    lambda_tvl = 1e-4
    tvl = tv_loss(recon_x)
    return mse + (beta * kld) + (lambda_tvl * tvl)

def lr_scheduler_func(epoch, num_epochs, warmup_epochs=5, min_lr=1e-4):
    if epoch < warmup_epochs: return float(epoch / warmup_epochs)
    else: return min_lr + 0.5 * float(1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    # return 1e-2

def train_vae():
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE().to(device)
    patience = 500
    early_stopping_patirnce = 0
    # optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = optim.Adam(vae.parameters(), lr=5e-4)
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, cooldown=5)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    num_epochs = 50 * 10
    config = None
    scheduler = LambdaLR(optimizer, lr_lambda= lambda epoch: lr_scheduler_func(epoch, num_epochs))
    best_loss = torch.inf
    for epoch in range(num_epochs):
        epoch_loss = 0
        for triplanes in tqdm(triplane_dataloader(), desc=f'Epoch {epoch + 1} / {num_epochs} - Training'):
        # for triplanes in triplane_dataloader():
            optimizer.zero_grad()
            triplanes = triplanes.to(device)
            triplanes = triplanes.squeeze()
            recon_triplane, mu, logvar = vae(triplanes)
            loss = vae_loss(recon_triplane, triplanes, mu, logvar, epoch, num_epochs)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        loss_avg = epoch_loss / len(triplane_dataloader())
        # scheduler.step(loss_avg)
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {loss_avg}, LR: {scheduler.get_last_lr()}')
        if best_loss > loss_avg:
            best_loss = loss_avg
            config = {
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_avg,
            }
            early_stopping_patirnce = 0
        else:
            early_stopping_patirnce += 1
            if early_stopping_patirnce == patience: break
        if (epoch + 1) % 10 == 0:
            torch.save(config, f'{save_path}/{weights_dir}.pth')

def load_vae_checkpoint():
    vae = VAE().to(device)
    checkpoint = torch.load(f"./vae_weights/{weights_dir}.pth")
    optimizer = optim.Adam(vae.parameters(), lr=1e-2, betas=(0.5, 0.999))
    num_epochs = 50 * 20
    scheduler = LambdaLR(optimizer, lr_lambda= lambda epoch: lr_scheduler_func(epoch, num_epochs))
    vae.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_avg = checkpoint['loss_avg']

    print(f"Resuming training from epoch {start_epoch}, last average loss: {loss_avg:.6f}")

def save_latent_representation():
    vae  = VAE().to(device)
    vae_weights_dir = f'./vae_weights/{weights_dir}.pth'
    checkpoint = torch.load(vae_weights_dir)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    latent_output_dir = './latent_images'
    os.makedirs(latent_output_dir, exist_ok=True)
    with torch.no_grad():
        for i, triplanes in enumerate(triplane_dataloader()):
            triplanes = triplanes.to(device)
            triplanes = triplanes.squeeze()
            mu, logvar = vae.encode(triplanes)
            # latent_representation = torch.cat([mu, logvar], dim=1)
            latent_path = os.path.join(latent_output_dir, f'latent_{i+1}.pt')
            torch.save(mu.cpu(), latent_path)    # latent shape: batch_size * num_planes(3) x num_features(3) x height(32) x width(32)
            z_reparametrized = vae.reparameterize(mu, logvar)
            z_decoded = vae.decode(z_reparametrized)
            z_decoded = z_decoded.cpu().permute(0, 2, 3, 1).contiguous().numpy()
            # if i > 99:
            viz_projections(z_decoded[0], z_decoded[1], z_decoded[2])

def main():
    print('Main function: VAE')
    train_vae()
    save_latent_representation()

if __name__ == '__main__':
    main()
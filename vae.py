import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import triplane_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
triplane_data = triplane_dataloader()
latent_dimension = 32
mean_logvar_split = 3
# logvar_split = 2
num_channels = 4

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            # nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            nn.Dropout(0.2),
            # nn.Conv2d(16, num_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, num_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(mean_logvar_split, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(16, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        mu = x[:, :mean_logvar_split, :, :]  # first 2 channels for mean
        logvar = x[:, mean_logvar_split:, :, :]  # last 2 channels for logvar
        return mu, logvar
        # return x.view(-1, 3, 32, 32, 3)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder_conv(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def tv_loss(x):
    batch_size, channels, h_x, w_x = x.size()
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    tv_h = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    tv_h = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    return (tv_h / count_h + tv_h / count_w) / batch_size

def vae_loss(recon_x, x, mu, logvar, beta=2e-3):
    # BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    TV = tv_loss(recon_x)
    L1 = torch.norm(x - recon_x, p=torch.inf)
    return BCE + KLD + beta * TV + L1
    # return BCE + KLD + beta * TV

def train_vae(ae=None):
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE().to(device) if ae==None else ae
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    num_epochs = 100

    for epoch in range(num_epochs):
        epoch_loss = 0
        for triplanes in triplane_data:
            triplanes = triplanes.to(device)
            batch_size, num_planes, height, width, channels = triplanes.size()
            triplanes = triplanes.view(batch_size * num_planes, height, width, channels)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(triplanes)
            loss = vae_loss(recon_batch, triplanes, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(triplane_data)}')
        scheduler.step(epoch_loss / len(triplane_data))
        # Save model weights
    torch.save(vae.state_dict(), f'{save_path}/weights.pth')

def viz_projections(xy_projection, yz_projection, zx_projection):
    # Visualizing projections
    cmap = 'viridis'   # Choosing 'gray' or 'viridis'
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('XY Projection')
    plt.imshow(xy_projection, cmap=cmap)
    plt.subplot(1, 3, 2)
    plt.title('YZ Projection')
    plt.imshow(yz_projection, cmap=cmap)
    plt.subplot(1, 3, 3)
    plt.title('ZX Projection')
    plt.imshow(zx_projection, cmap=cmap)
    plt.show() 

def save_latent_representation(dataloader, vae, output_dir):
    vae.eval()  # Set the VAE to evaluation mode
    os.makedirs(output_dir, exist_ok=True)
    latent_records = []
    with torch.no_grad():
        for i, triplanes in enumerate(dataloader):
            triplanes = triplanes.to(device)  # Moving data to the appropriate device
            # Reshape triplanes to (batch_size * num_planes, height, width, channels)
            triplanes = triplanes.permute(0, 1, 4, 2, 3).contiguous().view(-1, num_channels, 128, 128)
            mu, logvar = vae.encode(triplanes)
            latent_representation = torch.cat([mu, logvar], dim=1)
            latent_path = os.path.join(output_dir, f'latent_{i}.pt')
            torch.save(latent_representation.cpu(), latent_path)    # latent shape: batch_size * num_planes(3) x num_features(4) x height(32) x width(32)
            latent_representation = latent_representation[:, :3, :, :].permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
            viz_projections(latent_representation[0], latent_representation[1], latent_representation[2])

def main():
    print('Main function: VAE')
    # latent_output_dir = './latents'
    # vae = VAE().to(device)
    # train_vae()
    # save_latent_representation(triplane_data, vae, latent_output_dir)   # Saving the latent representations

if __name__ == '__main__':
    main()
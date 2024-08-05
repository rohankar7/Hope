import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import triplane_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
triplane_data = triplane_dataloader()

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder
        feature_dim = 256 * 8 * 8
        # feature_dim = 4 * 2 * 2
        num_channels = 1
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU()
        )
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, feature_dim)
        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(1, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 256, 8, 8)
        # x = x.view(x.size(0), 4, 2, 2)
        return self.decoder_conv(x)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def tv_loss(x):
    batch_size = x.size(0)
    h_x = x.size(2)
    w_x = x.size(3)
    count_h = (x.size(2) - 1) * x.size(3)
    count_w = x.size(2) * (x.size(3) - 1)
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def vae_loss(recon_x, x, mu, logvar, beta=2e-3):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    TV = tv_loss(recon_x)
    return BCE + KLD + beta * TV

def train_vae(ae=None):
    latent_dim = 64
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE(latent_dim).to(device) if ae==None else ae
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    num_epochs = 100

    for epoch in range(num_epochs):
        epoch_loss = 0
        for triplanes in triplane_data:
            triplanes = triplanes.to(device)
            batch_size, num_planes, channels, height, width = triplanes.size()
            triplanes = triplanes.view(batch_size * num_planes, channels, height, width)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(triplanes)
            loss = vae_loss(recon_batch, triplanes, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(triplane_data)}')
        # scheduler.step(epoch_loss / len(triplane_data))
        # Save model weights
    torch.save(vae.state_dict(), f'{save_path}/weights.pth')

def save_latent_representation(dataloader, vae, output_dir):
    vae.eval()  # Set the VAE to evaluation mode
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, triplanes in enumerate(dataloader):
            triplanes = triplanes.to(device)  # Move data to the appropriate device
            batch_size, num_planes, channels, height, width = triplanes.size()
            
            # Reshape triplanes to (batch_size * num_planes, channels, height, width)
            triplanes = triplanes.view(batch_size * num_planes, channels, height, width)
            mu, logvar = vae.encode(triplanes)
            latent_representation = vae.reparameterize(mu, logvar)            
            # Save each latent representation in the batch
            for j in range(batch_size * num_planes):
                # latent_path = os.path.join(output_dir, f'latent_{i * batch_size * num_planes + j}.pt')
                # TODO: Maintain industry standards and replace the loc below
                latent_path = os.path.join(output_dir, f'latent_{i}.pt')
                torch.save(latent_representation[j].cpu(), latent_path)

def main():
    latent_output_dir = './latents'
    vae = VAE(latent_dim=64).to(device)
    # Training the vae
    train_vae()
    # Save latent representations
    # vae.load_state_dict(torch.load('./vae_weights/weights.pth'))
    save_latent_representation(triplane_data, vae, latent_output_dir)

if __name__ == '__main__':
    main()
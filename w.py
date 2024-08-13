import torch
from torch import nn, optim
import torch.nn.functional as F
from data_loader import triplane_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
triplane_data = triplane_dataloader()
# If the resolution of the triplane data changes from 128 to 256, add layers to the encoder and decoder
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_channels = 1
        self.latent_height = self.latent_width = latent_dim
        # Encoder
        feature_dim = 256 * 2 * 2 # Expected: 32 * 256 * 256
        # feature_dim = 4 * 2 * 2
        latent_size = 1 * latent_dim * latent_dim # Expected: 8 * 32 * 32
        num_features = 1 # Expected: 32
        self.encoder_conv = nn.Sequential(
            # nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU()
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            # nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(num_features),
            # nn.ReLU()
        )
        self.fc_mu = nn.Linear(feature_dim, latent_size)
        self.fc_logvar = nn.Linear(feature_dim, latent_size)
        # Decoder
        self.decoder_fc = nn.Linear(latent_size, feature_dim)
        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(num_features, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        l = x
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return l, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), -1).unsqueeze(1).reshape(x.size(0), 1, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))) # Reshape to (batch_size, channels, 4, 4)
        return self.decoder_conv(x)
    
    def forward(self, x):
        l, mu, logvar = self.encode(x)
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

def vae_loss(recon_x, x, mu, logvar, beta=2e-3, kl_weight = 1e-5):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    TV = tv_loss(recon_x)
    return BCE + KLD * kl_weight + beta * TV

def train_vae(ae=None):
    latent_dim = 32
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE(latent_dim).to(device) if ae==None else ae
    vae.train()
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    num_epochs = 100
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        for triplanes in triplane_data:
            triplanes = triplanes.to(device)
            batch_size, num_planes, channels, height, width = triplanes.size()
            triplanes = triplanes.view(batch_size * num_planes, channels, height, width)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(triplanes)
            loss = vae_loss(recon_batch, triplanes, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0) # What does this do?
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
            triplanes = triplanes.to(device)
            batch_size, num_planes, channels, height, width = triplanes.size()
            # Reshape triplanes to (batch_size * num_planes, channels, height, width)
            triplanes = triplanes.view(batch_size * num_planes, channels, height, width)
            l, mu, logvar = vae.encode(triplanes)
            latent_representation = vae.reparameterize(mu, logvar) # latent_representation.shape =  torch.Size([3, 1024])
            # Save each latent representation in the batch
            for j in range(batch_size * num_planes):
                # latent_path = os.path.join(output_dir, f'latent_{i * batch_size * num_planes + j}.pt')
                # TODO: Maintain industry standards and replace the loc below
                latent_path = os.path.join(output_dir, f'latent_{i}.pt')
                # torch.save(latent_representation[j].cpu(), latent_path)
                torch.save(l.cpu(), latent_path)

def main():
    latent_output_dir = './latents'
    vae = VAE(latent_dim=32).to(device)
    # Training the vae
    train_vae() # Uncomment while training
    # vae.load_state_dict(torch.load('./vae_weights/weights.pth')) # Load vae_weights if not training
    # Save latent representations
    save_latent_representation(triplane_data, vae, latent_output_dir)

if __name__ == '__main__':
    main()
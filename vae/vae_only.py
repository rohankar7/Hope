import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=4, stride=2, padding=1),  # (12, 256, 256) -> (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 128, 128) -> (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128, 64, 64) -> (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (256, 32, 32) -> (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # (512, 16, 16) -> (1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (1024, 8, 8) -> (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (512, 16, 16) -> (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (256, 32, 32) -> (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 64, 64) -> (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 12, kernel_size=4, stride=2, padding=1),  # (64, 128, 128) -> (12, 256, 256)
            nn.Sigmoid(),  # Constrain output to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 8, 8)  # Reshape to decoder input dimensions
        x = self.decoder(x)

        return x, mu, logvar

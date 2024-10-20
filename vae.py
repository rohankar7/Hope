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
mean_logvar_split = 2
# logvar_split = 2
num_channels = 3

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Check z.py for loss BCE or MSE?
        # Encoder
        self.encoder_conv = nn.Sequential(
            # nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_channels, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            # nn.Dropout(0.2),
            # nn.Conv2d(16, num_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
        self.flattened_dim = 32 * latent_dimension * latent_dimension
        self.fc1 = nn.Linear(self.flattened_dim, 3 * latent_dimension * latent_dimension)
        # self.fc2 = nn.Linear(256, latent_dimension)
        self.dc1 = nn.Linear(num_channels * latent_dimension * latent_dimension, self.flattened_dim)
        # Decoder
        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(16, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )
    
    def encode(self, x):
        # x = self.encoder_conv(x)
        # mu = x[:, :mean_logvar_split, :, :]  # first 2 channels for mean
        # logvar = x[:, mean_logvar_split:, :, :]  # last 2 channels for logvar
        # return mu, logvar
        # return x.view(-1, 3, 32, 32, 3)
        x = self.encoder_conv(x)
        x = x.view(-1, self.flattened_dim)
        return self.fc1(x), self.fc1(x) # For mu and logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # return mu + eps * std
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = self.dc1(z)
        z = z.reshape(3, 32, latent_dimension, latent_dimension)
        return self.decoder_conv(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        # mu, logvar = mu.squeeze(), logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def tv_loss(x):
    batch_size, channels, h_x, w_x = x.size()
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    tv_h = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    tv_h = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    return (tv_h / count_h + tv_h / count_w) / batch_size

def vae_loss(recon_x, x, mu, logvar, beta=1e-4):
    # a = recon_x.cpu().permute(0, 2, 3, 1).detach().numpy()
    # viz_projections(a[0], a[1], a[2])
    # b = x.cpu().permute(0, 2, 3, 1).detach().numpy()
    # viz_projections(b[0], b[1], b[2])
    # BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    mse = F.mse_loss(recon_x, x, reduction='mean')
    # KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    kld = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    tvl = tv_loss(recon_x)
    # l1_norm = torch.norm(x - recon_x, p=1) / x.size(0)
    l1_norm = torch.norm(x - recon_x, p=1)
    # return MSE + KLD * 0.1
    # print('Losses', mse.cpu().detach().numpy() , kld.cpu().detach().numpy() , tvl.cpu().detach().numpy(), l1_norm.cpu().detach().numpy())
    return mse + kld + beta * (tvl)
    # return mse + kld + beta * (tvl + l1_norm)

def train_vae():
    # save_path = './vae_weights'
    save_path = './weights'
    os.makedirs(save_path, exist_ok=True)
    vae = VAE().to(device)
    # optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_rate = 2e-4
    # optimizer = optim.Adam(vae.parameters(), lr=lr_rate)
    optimizer = optim.Adam(vae.parameters(), lr=lr_rate, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for triplanes in triplane_data:
            triplanes = triplanes.to(device)
            batch_size, num_planes, channels, height, width = triplanes.size()
            # print('Hey', batch_size, num_channels, channels, height, width)
            triplanes = triplanes.view(batch_size * num_planes, channels, height, width)
            optimizer.zero_grad()
            recon_triplane, mu, logvar = vae(triplanes)
            loss = vae_loss(recon_triplane, triplanes, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        loss_avg = epoch_loss / len(triplane_data)
        scheduler.step(loss_avg)
        print(f'Epoch {epoch}, Loss: {loss_avg}')
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

def save_latent_representation():
    vae  = VAE().to(device)
    # vae_weights_dir = './vae_weights/weights.pth'
    vae_weights_dir = './weights/weights.pth'
    vae.load_state_dict(torch.load(vae_weights_dir))
    vae.eval()  # Set the VAE to evaluation mode
    latent_output_dir = './latents'
    os.makedirs(latent_output_dir, exist_ok=True)
    latent_encodings = []
    with torch.no_grad():
        for i, triplanes in enumerate(triplane_dataloader()):
            triplanes = triplanes.to(device)  # Moving data to the appropriate device
            # triplanes = triplanes.permute(0, 1, 4, 2, 3).contiguous().view(-1, num_channels, 256, 256)
            b, p, c, h, w = triplanes.size()
            triplanes = triplanes.view(b*p, c, h, w)
            mu, logvar = vae.encode(triplanes)
            latent_representation = torch.cat([mu, logvar], dim=1)
            latent_path = os.path.join(latent_output_dir, f'latent_{i}.pt')
            latent_encodings.append((mu, logvar))
            torch.save(latent_representation.cpu(), latent_path)    # latent shape: batch_size * num_planes(3) x num_features(3) x height(32) x width(32)
        z_decoded = None
        for _ in latent_encodings[:]:
            z_reparametrized = vae.reparameterize(mu, logvar)
            z_decoded = vae.decode(z_reparametrized)
            # print(z_decoded.shape)
            z_decoded = z_decoded.cpu().permute(0, 2, 3, 1).numpy()
            # print(z_decoded)
            viz_projections(z_decoded[0], z_decoded[1], z_decoded[2])

def main():
    print('Main function: VAE')
    train_vae()
    save_latent_representation()   # Saving the latent representations
    # model = VAE()
    # print(model)

if __name__ == '__main__':
    main()
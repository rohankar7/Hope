import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from data_loader import triplane_dataloader
import matplotlib.pyplot as plt
import os

# Input image -> Hidden Image -> mean, std -> Parametrization trick -> Decoder -> Output image
class VAE(nn.Module):
    def __init__(self, input_dim=3*256*256, hidden_dim = 200, latent_dim = 3*32*32):
        super().__init__()
        # Encoder
        self.img_2hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, latent_dim)
        self. hid_2sigma = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.z_2hid = nn.Linear(latent_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        z_reconstructed = self.decode(z_reparametrized)
        return z_reconstructed, mu, sigma
    
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

def train():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './vae_weights'
    os.makedirs(save_path, exist_ok=True)
    input_dim = 256*256*3
    hidden_dim = 200
    latent_dim = 64*64*3
    num_epochs = 3
    lr_rate = 1e-4
    triplane_data = triplane_dataloader()
    vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr_rate)
    loss_func = nn.BCELoss(reduction='sum')
    # Training loop
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(triplane_data))
        for i, (_, triplane) in enumerate(loop):
            b, p, c, h, w = triplane.size()
            triplane = triplane.to(device).view(b*p, c*h*w)
            triplane_reconstructed, mu, sigma = vae(triplane)
            # Compute loss
            reconstruction_loss = loss_func(triplane_reconstructed, triplane)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item())
    torch.save(vae.state_dict(), f'{save_path}/weights.pth')
    vae.eval()
    os.makedirs('./latent_images', exist_ok=True)
    encodings = []
    with torch.no_grad():
        for i, triplane in enumerate(triplane_dataloader()):
            # Reshape triplanes to (batch_size * num_planes, height, width, channels)
            b, p, c, h, w = triplane.size()
            triplane = triplane.to('cuda').view(b*p, c*h*w)
            mu, sigma = vae.encode(triplane)
            encodings.append((mu, sigma))
        for i in range(len(encodings)):
            epsilon = torch.rand_like(sigma)
            z = mu + sigma * epsilon
            out = vae.decode(z)
            out = out.view(3, 3, 256, 256).permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
            x = out[0]
            y = out[1]
            z = out[2]
            viz_projections(x, y, z)

def save_latent_representation(output_dir='./latent_images'):
    vae  = VAE().to('cuda')
    vae.load_state_dict(torch.load('./vae_weights/weights.pth'))
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)
    encodings = []
    with torch.no_grad():
        for i, triplanes in enumerate(triplane_dataloader()):
            # Reshape triplanes to (batch_size * num_planes, height, width, channels)
            b, p, c, h, w = triplane.size()
            triplane = triplane.to('cuda').view(b*p, c*h*w)
            mu, sigma = vae.encode(triplanes)
            encodings.append((mu, sigma))
        for i in range(len(encodings)):
            epsilon = torch.rand_like(sigma)
            z = mu + sigma * epsilon
            out = vae.decode(z)
            out = out.view(3, 3, 256, 256)
            viz_projections(out[0], out[1], out[2])

def main():
    train()
    # x = torch.randn(4,256*256)  # Batch size = 4, Triplane resolution = 256
    # vae = VAE(input_dim = 256*256)
    # x_reconstructed, mu, sigma = vae(x)
    # print(x_reconstructed.shape)
    # print(mu.shape)
    # print(sigma.shape)
    # save_latent_representation()

if __name__ == '__main__':
    main()
import torch
from data_loader import latent_dataloader, embedding_dataloader

class NoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

    def get_noise_level(self, t):
        return self.betas[t]

import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention = self.softmax(attention_scores)
        attended_values = torch.bmm(attention, v)
        return attended_values

class UNet(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                    nn.Sequential(
                        nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                        nn.BatchNorm2d(feature),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                        nn.BatchNorm2d(feature),
                        nn.ReLU(inplace=True)
                    )
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(features[0], 4, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = self.ups[idx][0](x)
            skip_connection = skip_connections[idx]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx][1](x)
        return self.final_conv(x)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader



# Assume dataloader is already defined
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
def train():
    # Model, optimizer, and loss setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = NoiseScheduler()
    criterion = nn.MSELoss()
    dataloader = latent_dataloader()
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for idx, data in enumerate(dataloader):
            b, p, c, h, w = data.size()
            data = data.view(b*p, c, h, w)
            data = data.to(device)  # Assuming data is directly the input without labels
            print(data.shape)

            # Sample timestep t
            t = torch.randint(0, scheduler.num_timesteps, (1,)).item()
            noise_level = scheduler.get_noise_level(t)
            noisy_data = data + torch.randn_like(data) * noise_level  # Adding noise
            optimizer.zero_grad()
            # Predict noise (reverse process)
            predicted_noise = model(noisy_data)
            # Calculate loss (could vary depending on exact goal, here predicting noise directly)
            loss = criterion(predicted_noise, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# Running the training loop
train()
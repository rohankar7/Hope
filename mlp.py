import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data_loader import voxel_dataloader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def plot_voxel(voxel_data):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_data, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

class TriplaneMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.mlp_layers(x)
        return x.view(-1, 64, 64, 64)

def train_val_mlp():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 3*128*128*3
    output_dim = 64*64*64
    model = TriplaneMLP(input_dim, output_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    voxel_train_dataloader, voxel_val_dataloader = voxel_dataloader()
    weights_dir = './mlp_weights'
    os.makedirs(weights_dir, exist_ok=True)
    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for features, targets in voxel_train_dataloader:
            optimizer.zero_grad()
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        epoch_train_loss = training_loss / len(voxel_train_dataloader)
        print(f'Epoch {epoch+1}, Training loss: {epoch_train_loss:.4f}')

        # Validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for features, targets in voxel_val_dataloader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()
        epoch_val_loss = validation_loss / len(voxel_val_dataloader)
        print(f'Epoch {epoch+1}, Validation loss: {epoch_val_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{weights_dir}/mlp_weights_{epoch+1}.pth')

def main():
    train_val_mlp()

if __name__ == '__main__':
    main()
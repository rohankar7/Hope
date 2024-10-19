import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data_loader import voxel_dataloader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import config

# TODO: Integrate recommended updates and optimizations
# TODO: Include TV loss and L1 loss. Together with L2 loss between ground truth (Gc)
voxel_res = config.voxel_resolution
triplane_res = config.triplane_resolution

class TriplaneMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden_dim1 = 512
        self.hidden_dim2 = 256
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.mlp_layers(x)
        if config.voxel_type == 'color':    
            return x.view(-1, voxel_res, voxel_res, voxel_res, 3)
        return x.view(-1, voxel_res, voxel_res, voxel_res)

def train_val_mlp():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 3 * triplane_res * triplane_res * config.triplane_features
    output_dim = voxel_res * voxel_res * voxel_res
    if config.voxel_type == 'color':
        output_dim *= 3
    model = TriplaneMLP(input_dim, output_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    voxel_train_dataloader, voxel_val_dataloader = voxel_dataloader()
    os.makedirs(config.mlp_weights_dir, exist_ok=True)
    early_stopping_counter = 0
    save_interval = 10
    early_stopping_patience = 100
    best_val_loss = float('inf')
    # Training
    num_epochs = 100
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0
        for features, targets in tqdm(voxel_train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            features = features.to(device)
            targets = targets.to(device)
            if config.voxel_type == 'color':
                targets = targets / 255
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        epoch_train_loss = training_loss / len(voxel_train_dataloader)
        print(f'Epoch {epoch+1}, Training loss: {epoch_train_loss:.2f}')
        # Validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for features, targets in tqdm(voxel_val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                features = features.to(device)
                targets = targets.to(device)
                if config.voxel_type == 'color':
                    targets = targets / 255
                outputs = model(features)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()
        epoch_val_loss = validation_loss / len(voxel_val_dataloader)
        print(f'Epoch {epoch+1}, Validation loss: {epoch_val_loss:.2f}')
        # scheduler.step()
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print('Saving the model checkpoints at best validation loss:', best_val_loss)
            torch.save(model.state_dict(), f'{config.mlp_weights_dir}/best_mlp_weights.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'{config.mlp_weights_dir}/mlp_weights_{epoch+1}.pth')
        torch.cuda.empty_cache()

def main():
    train_val_mlp()

if __name__ == '__main__':
    main()
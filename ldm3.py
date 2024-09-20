# import torch
# from torch import nn, optim
# import torch.nn.functional as F
# from data_loader import latent_dataloader
# import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # Define the device

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class LatentDiffusionModel(nn.Module):
#     def __init__(self, latent_dim=64):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, latent_dim)
#         )
        
#     def forward(self, x):
#         return self.model(x)

# # Instantiate the model and move it to the device
# latent_dim = 64
# ldm = LatentDiffusionModel(latent_dim).to(device)
# optimizer = optim.Adam(ldm.parameters(), lr=1e-4)

# # Function to save a checkpoint
# def save_checkpoint(model, optimizer, epoch, path):
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, path)

# # Training loop
# num_epochs = 100
# checkpoint_dir = './ldm_checkpoints'

# for epoch in range(num_epochs):
#     epoch_loss = 0
#     dataloader = latent_dataloader()
#     for batch in dataloader:  # Assuming you have a dataloader for your latent space data
#         latent_vectors = batch.to(device)  # Your batch of latent vectors
        
#         optimizer.zero_grad()
#         output = ldm(latent_vectors)
#         loss = nn.MSELoss()(output, latent_vectors)  # Example loss function
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
    
#     print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}')
    
#     # Save checkpoint every 10 epochs
#     if epoch % 10 == 0:
#         checkpoint_path = f'{checkpoint_dir}/ldm_epoch_{epoch}.pth'
#         save_checkpoint(ldm, optimizer, epoch, checkpoint_path)
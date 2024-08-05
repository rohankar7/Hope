import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os

class TriplaneDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        triplane_data = np.load(file_path)  # Load the numpy stack
        xy_plane, yz_plane, zx_plane = None, None, None
        if self.transform:
            xy_plane = self.transform(triplane_data[0])
            yz_plane = self.transform(triplane_data[1])
            zx_plane = self.transform(triplane_data[2])
        return torch.stack([xy_plane, yz_plane, zx_plane], dim=0)

def triplane_dataloader(): # Storing triplane paths in a list
    out_dir = './images'
    triplane_paths = [os.path.join(out_dir, path) for path in os.listdir(out_dir)]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TriplaneDataset(triplane_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # (batch_size, 3, 1, 128, 128)
    return dataloader

class LatentDataset(Dataset):
    def __init__(self, latent_file_paths):
        self.latent_file_paths = latent_file_paths
    def __len__(self):
        return len(self.latent_file_paths)
    def __getitem__(self, idx):
        latent_path = self.latent_file_paths[idx]
        return torch.load(latent_path)

def latent_dataloader(): # Storing latent paths in a list
    latent_dir = './latents'
    latents = [f'{latent_dir}/{latents}' for latents in os.listdir(latent_dir)]
    dataset = LatentDataset(latents)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader
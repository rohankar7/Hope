import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os
shuffle_condition = False
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
        transforms.Normalize(mean = 0, std = 1),
    ])
    dataset = TriplaneDataset(triplane_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle_condition)
    return dataloader # Returns triplanes in the shape of (batch_size, 3, 1, 128, 128)

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle_condition)
    return dataloader # Returns latents in the shape of (batch_size, 3, 1, 32, 32)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_df):
        self.embeddings = embeddings_df['Embedding'].apply(eval).apply(torch.tensor)
        # self.labels = torch.tensor(embeddings_df['Subclass'].values, dtype=torch.long)
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx]
        # return {
        #     'embedding': self.embeddings[idx],
        #     'label': self.labels[idx]
        # }
    
def embedding_dataloader(): # Storing latent paths in a list
    torch.set_printoptions(precision=10)
    embeddings_dir = './text/embedding.csv'
    embeddings_df = pd.read_csv(embeddings_dir)  # Replace with your file path
    dataset = EmbeddingDataset(embeddings_df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle_condition)
    return dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
shuffle_condition = False
import config
from sklearn.model_selection import train_test_split

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
    data_dir = f'./triplane_images_{config.triplane_resolution}'
    triplane_paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir)[:10]]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean = 0, std = 1),
    ])
    dataset = TriplaneDataset(triplane_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle_condition)
    return dataloader # Returns triplanes in the shape of (batch_size, 3, 1, 128, 128)

# Create custom Dataset
class TriplaneVoxelDataset(Dataset):
    def __init__(self, image_paths, voxel_paths):
        self.image_paths = image_paths
        self.voxel_paths = voxel_paths
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        # TODO: flatten the image
        triplane_image = np.load(self.image_paths[idx]).astype(np.float32) 
        flattened_image = torch.tensor(triplane_image.reshape(3 * 128 * 128 * 3), dtype=torch.float32)
        voxel_grid = np.load(self.voxel_paths[idx]).astype(np.float32)
        voxel_grid = torch.tensor(voxel_grid, dtype=torch.float32)
        return flattened_image, voxel_grid

def voxel_dataloader():
    voxel_dir = './voxel_data'
    voxel_paths = [os.path.join(voxel_dir, path) for path in os.listdir(voxel_dir)[:10]  if path.endswith('.npy')]
    triplane_dir = f'./triplane_images_{config.triplane_resolution}'
    triplane_paths = [os.path.join(triplane_dir, path) for path in os.listdir(triplane_dir)[:10]]
    train_img_files, valid_img_files, train_voxel_files, valid_voxel_files = train_test_split(triplane_paths, voxel_paths, test_size=0.2, random_state=42)
    train_dataset = TriplaneVoxelDataset(train_img_files, train_voxel_files)
    valid_dataset = TriplaneVoxelDataset(valid_img_files, valid_voxel_files)
    voxel_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle_condition)
    voxel_val_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=shuffle_condition)
    return voxel_train_dataloader, voxel_val_dataloader

class LatentDataset(Dataset):
    def __init__(self, latent_file_paths):
        self.latent_file_paths = latent_file_paths
    def __len__(self):
        return len(self.latent_file_paths)
    def __getitem__(self, idx):
        latent_path = self.latent_file_paths[idx]
        return torch.load(latent_path)

def latent_dataloader(): # Storing latent paths in a list
    latent_dir = f'./latent_images_{config.triplane_resolution}'
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
    triplane_models = [x.split('.')[0].split('_')[1] for x in os.listdir('./images')]
    embeddings_df = embeddings_df[embeddings_df['Subclass'].isin(triplane_models)]
    dataset = EmbeddingDataset(embeddings_df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle_condition)
    return dataloader
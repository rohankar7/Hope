import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import measure

# Load the .obj file
def load_mesh(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    return mesh

# Compute the Signed Distance Field
def compute_sdf(mesh, grid_size=20):
    bounds = mesh.bounds
    span = np.linspace(bounds[0], bounds[1], num=grid_size)
    x, y, z = np.meshgrid(span[:, 0], span[:, 1], span[:, 2], indexing='ij')
    coords = np.stack([x, y, z], axis=-1)
    sdf = trimesh.proximity.signed_distance(mesh, coords.reshape(-1, 3))
    sdf = sdf.reshape(grid_size, grid_size, grid_size)
    return sdf

def generate_coordinate_grid(grid_size, device):
    lin = torch.linspace(-1, 1, steps=grid_size, device=device)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
    coords = torch.stack((x, y, z), dim=-1)  # Stack into coordinate vectors
    coords = coords.unsqueeze(0)  # Add batch dimension
    coords = coords.permute(0, 4, 1, 2, 3)  # Rearrange to [batch, channels, depth, height, width]
    return coords

# Neural network definition
class TriPlane(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Each plane gets a separate 2D convolution
        self.xy = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.yz = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.zx = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, channels, depth, height, width]
        # Expected shape for Conv2d is [batch, channels, height, width]
        xy_plane = x[:, :, :, :, :].mean(dim=2)  # Average over depth to collapse into 2D
        yz_plane = x[:, :, :, :, :].mean(dim=4)  # Average over width
        zx_plane = x[:, :, :, :, :].mean(dim=3)  # Average over height

        xy_features = self.xy(xy_plane)
        yz_features = self.yz(yz_plane)
        zx_features = self.zx(zx_plane)

        return xy_features, yz_features, zx_features

class HybridNeuralField(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()
        self.triplane = TriPlane(3, feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, coords):
        xy_feat, yz_feat, zx_feat = self.triplane(coords)
        # Flattening and concatenating features from all planes
        features = torch.cat([xy_feat.flatten(start_dim=1), yz_feat.flatten(start_dim=1), zx_feat.flatten(start_dim=1)], dim=1)
        sdf = self.mlp(features)
        return sdf

# Reconstruct the 3D model using marching cubes
def reconstruct_mesh(sdf, level=0):
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return reconstructed_mesh

# Main training and execution flow
def main(file_path):
    mesh = load_mesh(file_path)
    sdf = compute_sdf(mesh)

    sdf_tensor = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
    coords = generate_coordinate_grid(grid_size=sdf_tensor.shape[-1], device='cpu')

    model = HybridNeuralField(feature_dim=64, hidden_dim=128, out_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(coords)
        loss = nn.functional.mse_loss(output, sdf_tensor.view(output.size()))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Reconstruct the mesh from the SDF
    output_sdf = output.detach().numpy().squeeze()  # Removing batch dimension
    reconstructed_mesh = reconstruct_mesh(output_sdf)
    reconstructed_mesh.show()

if __name__ == "__main__":
    main("C:/ShapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj")

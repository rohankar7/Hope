import torch
import torch.nn as nn
import trimesh
import numpy as np
from skimage import measure

# Load and prepare the mesh
def load_mesh(file_path):
    return trimesh.load(file_path, force='mesh')

# Create the signed distance field
def compute_sdf(mesh, grid_size=64):
    bounds = mesh.bounds
    span = np.linspace(bounds[0], bounds[1], num=grid_size).astype(np.float32)
    x, y, z = np.meshgrid(span[:, 0], span[:, 1], span[:, 2], indexing='ij')
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    coords = np.stack([x, y, z], axis=-1)
    sdf = trimesh.proximity.signed_distance(mesh, coords.reshape(-1, 3))
    sdf = sdf.astype(np.float32).reshape(grid_size, grid_size, grid_size)
    return sdf, span

# Define the neural network architecture
class HybridNeuralField(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Using a simple MLP for illustration
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Reconstruct the mesh using marching cubes
def reconstruct_mesh(sdf, span, level=0):
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    vertices = vertices * (span[1] - span[0]) / len(sdf) + span[0]  # Rescale vertices
    reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return reconstructed_mesh

# Path to the OBJ file
file_path = "C:/ShapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj"


# Load the mesh
mesh = load_mesh(file_path)

# Compute the SDF
sdf, span = compute_sdf(mesh, grid_size=128)

# Convert SDF to a tensor
sdf_tensor = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

# Neural network setup
model = HybridNeuralField(feature_dim=3, hidden_dim=64, out_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Dummy coordinates for input (assuming a fixed grid size)
coords = np.indices(sdf.shape).reshape(3, -1).T / sdf.shape[0]
coords_tensor = torch.tensor(coords, dtype=torch.float32)

# Training loop
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    predictions = model(coords_tensor).reshape(sdf.shape)
    loss = criterion(predictions, sdf_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Using the trained model to generate a new SDF
with torch.no_grad():
    new_sdf = model(coords_tensor).reshape(sdf.shape).numpy()

# Reconstruct the mesh from the new SDF
reconstructed_mesh = reconstruct_mesh(new_sdf, span)
reconstructed_mesh.show()
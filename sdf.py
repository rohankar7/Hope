import torch
from torch import nn, optim
import trimesh
import numpy as np
import warnings
from skimage import measure

warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def train_model(sdf_tensor):
    # Neural network setup
    model = HybridNeuralField(feature_dim=3, hidden_dim=64, out_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(100):  # Number of epochs
        optimizer.zero_grad()
        output = model(sdf_tensor)
        loss = criterion(output, sdf_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def sample_points_around_mesh(mesh, grid_size=30, padding=0.1):
    """
    Create a grid of points around the mesh to compute the SDF.
    :param mesh: trimesh object.
    :param grid_size: resolution of the grid in each dimension.
    :param padding: extra space around the mesh bounds.
    """
    min_bound, max_bound = mesh.bounds  # Get the bounds of the mesh
    bounds_range = max_bound - min_bound
    padded_bounds = [min_bound - padding * bounds_range, max_bound + padding * bounds_range]    # Adding padding around the bounds
    print('Bounds', padded_bounds)
    # Generate a grid of points
    x = np.linspace(padded_bounds[0][0], padded_bounds[1][0], num=grid_size)
    y = np.linspace(padded_bounds[0][1], padded_bounds[1][1], num=grid_size)
    z = np.linspace(padded_bounds[0][2], padded_bounds[1][2], num=grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    return points, grid_x.shape

def compute_sdf(mesh, points):
    return trimesh.proximity.signed_distance(mesh, points)  # Compute the signed distance values from the mesh to the points

def load_watertight_mesh(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    # mesh.remove_degenerate_faces()  # Remove degenerate faces
    # mesh.merge_vertices()   # Merge vertices that are very close to each other
    # # Repair the mesh to ensure it is watertight if necessary
    # print(mesh.is_watertight)
    # if not mesh.is_watertight:
    #     print('Not watertight')
    #     mesh.fill_holes()
    return mesh

# Reconstruct the 3D model using marching cubes
def reconstruct_mesh(sdf, level=0):
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return reconstructed_mesh

def main(file_path, grid_size=20):
    mesh = load_watertight_mesh(file_path)
    points, shape = sample_points_around_mesh(mesh, grid_size)  # Points shape: 27000 x 3, Shape: 30 x 30 x 30
    sdf_values = compute_sdf(mesh, points)  # Getting the sdf values
    sdf_grid = sdf_values.reshape(shape)    # Resizing the sdf values to the desired shape
    sdf_tensor = torch.tensor(sdf_grid, dtype=torch.float32).unsqueeze(0)   # Adding the batch dimension
    mesh = reconstruct_mesh(sdf_tensor, level = 0)
    mesh.show()
    # train_model(sdf_tensor)

if __name__ == "__main__":
    file_path = "C:/ShapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj"
    sdf_grid = main(file_path)

    print(sdf_grid)


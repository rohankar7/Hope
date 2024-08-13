import numpy as np
import trimesh
import torch
from torch import optim
import os
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, disk
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
# from ldm import LatentDiffusionModel
from model import LatentDiffusionModel
from vae import *
from ShapeNetCore import *

def load_ldm_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def clean_projection(projection, operation='closing', radius=2):
    selem = disk(radius)
    if operation == 'closing':
        return binary_closing(projection, selem)
    elif operation == 'opening':
        return binary_opening(projection, selem)
    else:
        raise ValueError("Invalid operation. Use 'closing' or 'opening'.")
    # return projection

def smooth_voxel_grid(voxel_grid, iterations=1):
    for _ in range(iterations):
        voxel_grid = binary_dilation(voxel_grid)
        voxel_grid = binary_erosion(voxel_grid)
    return voxel_grid

def fill_voxel_grid(xy_projection, yz_projection, zx_projection, resolution):
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    for x in range(resolution):
        for y in range(resolution):
            if xy_projection[y, x]:  # If the pixel is part of the object in the XY plane
                for z in range(resolution):
                    if yz_projection[z, y] and zx_projection[z, x]:
                        voxel_grid[x, y, z] = 1
    return voxel_grid

def generate_mesh(xy_projection_cleaned, yz_projection_cleaned, zx_projection_rotated_cleaned, resolution):
    voxel_grid = fill_voxel_grid(xy_projection_cleaned, yz_projection_cleaned, zx_projection_rotated_cleaned, resolution)
    voxel_grid_smoothed = smooth_voxel_grid(voxel_grid, iterations=2)
            # Visualize the voxel grid
    threshold, min_value, max_value = None, voxel_grid.min(), voxel_grid.max()        
    if threshold is None: threshold = (min_value + max_value) / 2
    if not (min_value <= threshold <= max_value): # Ensure the threshold is within the data range
        raise ValueError(f"Threshold {threshold} is out of range ({min_value}, {max_value})")
    verts, faces, normals, values = measure.marching_cubes(voxel_grid_smoothed, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, vertex_attributes={'values':values})
    return mesh    

def repair_mesh(mesh):
    mesh.fill_holes()  # Fill holes in the mesh
    mesh.update_faces(mesh.nondegenerate_faces()) # Remove degenerate faces
    # mesh.update_faces(mesh.unique_faces())  # Remove duplicate faces
    mesh.remove_infinite_values()  # Remove infinite values
    mesh.remove_unreferenced_vertices()  # Remove unreferenced vertices
    mesh.update_faces(mesh.unique_faces())  # Remove duplicate vertices
    # mesh.show() # Visualize the repaired mesh
    return mesh

def model_from_triplanes(output_dir, resolution):
    for np_triplane in os.listdir(output_dir):
        print(np_triplane)
        triplane = np.load(os.path.join(output_dir, np_triplane))
        xy_projection, yz_projection, zx_projection_rotated = triplane[0], triplane[1], triplane[2]
        # Clean the projections
        xy_projection_cleaned = clean_projection(xy_projection, 'closing', 2)
        yz_projection_cleaned = clean_projection(yz_projection, 'closing', 2)
        zx_projection_rotated_cleaned = clean_projection(zx_projection_rotated, 'closing', 2)
        try:
            mesh = generate_mesh(xy_projection_cleaned, yz_projection_cleaned, zx_projection_rotated_cleaned, resolution)
        except RuntimeError as e:
            try:
                zx_projection_rotated = np.rot90(zx_projection_rotated, k=2)
                zx_projection_rotated_cleaned = clean_projection(zx_projection_rotated, 'closing', 2)
                mesh = generate_mesh(xy_projection_cleaned, yz_projection_cleaned, zx_projection_rotated_cleaned, resolution)
            except RuntimeError as e:
                try:
                    zx_projection_rotated = np.rot90(zx_projection_rotated, k=-1)
                    zx_projection_rotated_cleaned = clean_projection(zx_projection_rotated, 'closing', 2)
                    mesh = generate_mesh(xy_projection_cleaned, yz_projection_cleaned, zx_projection_rotated_cleaned, resolution)
                except RuntimeError as e:
                    print("No surface found at the given iso value")
        mesh = repair_mesh(mesh) # Mesh repairs
        model_gen_dir = './generated_models'
        os.makedirs(model_gen_dir, exist_ok=True)
        mesh.export(f'{model_gen_dir}/{np_triplane.split('.')[0]}.obj')
        print('Exported')

def main():
    # Instantiate the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 64
    ldm = LatentDiffusionModel(latent_dim).to(device)
    optimizer = optim.Adam(ldm.parameters(), lr=1e-4)

    # Load the checkpoint for ldm
    checkpoint_path = './ldm_checkpoints/ldm_epoch_90.pth'
    ldm, optimizer, start_epoch = load_ldm_checkpoint(ldm, optimizer, checkpoint_path)
    ldm.eval()

    # Decode latent triplanes
    vae  = VAE(latent_dim=latent_dim).to(device)
    # train_vae(vae)
    vae.load_state_dict(torch.load('./vae_weights/weights.pth'))
    vae.eval()
    latent_vectors = torch.load('./latents/latent_0.pt').to(device)
    print(latent_vectors.unsqueeze(0).shape)

    latent_triplanes = ldm(latent_vectors.unsqueeze(0))
    # latent_triplanes = ldm(latent_vectors)
    decoded_triplanes = vae.decode(latent_triplanes)
    print('Shape', decoded_triplanes.shape)
    model_from_triplanes('./images', resolution=128)

if __name__ == "__main__":
    main()
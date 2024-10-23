import trimesh
import numpy as np
import config
from ShapeNetCore import get_random_models
import config
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_voxel(voxel_data, threshold=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if config.voxel_type == 'color':
        if voxels.max() > 1:
            voxels = voxels / 255.0 # Normalizing the voxel colors for visualization
        mask = np.any(voxels > threshold, axis=-1) # Masking for non-zero voxels with color intensity > 0
        x, y, z = np.indices(voxels.shape[:-1])  # Getting the grid coordinates
        ax.scatter(x[mask], y[mask], z[mask], c=voxels[mask].reshape(-1, 3), marker='o', s=20)
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    else:
        ax.voxels(voxel_data, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def generate_colored_voxels(mesh, padded_voxels):
    voxel_res = config.voxel_resolution
    pitch_val = mesh.extents.max() / (voxel_res - 1)
    mesh.visual = mesh.visual.to_color()
        # if mesh.visual.kind == 'vertex':
    vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB values (ignore alpha if present)
    voxel_colors = np.zeros((voxel_res, voxel_res, voxel_res, 3), dtype=np.uint8)  # Initialize color grid
    for x in range(voxel_res):
        for y in range(voxel_res):
            for z in range(voxel_res):
                if padded_voxels[x, y, z]:
                    # voxel_position = np.array([x, y, z]) * pitch_val + mesh.bounds[0]
                    voxel_position = (np.array([x, y, z]) + 0.5) * pitch_val + mesh.bounds[0]
                    distances = np.linalg.norm(mesh.vertices - voxel_position, axis=1)
                    closest_vertex_idx = np.argmin(distances) # Find the nearest vertex from the original mesh
                    voxel_colors[x, y, z] = vertex_colors[closest_vertex_idx]

def create_voxel_grid():
    voxel_res = config.voxel_resolution
    os.makedirs(config.voxel_dir, exist_ok=True)
    for path in tqdm(sorted(os.listdir(config.triplane_dir)[:]), desc='Progress'):
        path = '/'.join(path.split('.')[0].split('_'))
        mesh_path = f'{config.pwd}/{path}/{config.suffix_dir}'
        file_name = '_'.join(path.split('/')) + '.npy'
        if file_name in os.listdir(config.voxel_dir): continue
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            pitch_val = mesh.extents.max() / (voxel_res - 1)
            voxel_grid = mesh.voxelized(pitch = pitch_val)
            voxel_data = voxel_grid.matrix
            padded_voxels = np.zeros((voxel_res, voxel_res, voxel_res), dtype=bool)
            offsets = [(voxel_res - min(s, voxel_res)) // 2 for s in voxel_data.shape]
            insert_slices = tuple(slice(offset, offset + min(s, voxel_res)) for offset, s in zip(offsets, voxel_data.shape))
            padded_voxels[insert_slices] = voxel_data[:voxel_res, :voxel_res, :voxel_res]
            if config.voxel_type == 'color':
                padded_voxels = generate_colored_voxels(mesh, padded_voxels)
            # visualize_voxel(padded_voxels) # Uncommenting this will display the generated coloured voxels
            np.save(f'{config.voxel_dir}/{file_name}', padded_voxels)
        except (IndexError, AttributeError, np.core._exceptions._ArrayMemoryError) as e:
            continue

def main():
    create_voxel_grid()

if __name__ == '__main__':
    main()
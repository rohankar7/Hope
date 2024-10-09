import trimesh
import numpy as np
import config
from ShapeNetCore import get_random_models
import config
import os
import matplotlib.pyplot as plt

def visualize_voxel(voxels):
    voxel_norm = voxels / 255.0 # Normalizing the voxel colors for visualization
    # Creating a mask for non-zero voxels with color intensity > 0
    mask = np.any(voxels > 0, axis=-1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Use scatter to plot the non-zero voxels with their corresponding colors
    x, y, z = np.indices(voxel_norm.shape[:-1])  # Get the grid coordinates
    ax.scatter(x[mask], y[mask], z[mask], c=voxel_norm[mask].reshape(-1, 3), marker='o', s=20)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    plt.show()

def create_voxel_grid():
    voxel_res = config.voxel_resolution
    output_dir = f'./voxel_data_{voxel_res}'
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(os.listdir(config.triplane_dir)[3:10]):
        path = '/'.join(path.split('.')[0].split('_'))
        mesh_path = f'{config.pwd}/{path}/{config.suffix_dir}'
        mesh = trimesh.load(mesh_path, force='mesh')
        # if mesh.visual.material:
        #     print(mesh.visual.material)
        pitch_val = mesh.extents.max() / (voxel_res - 1)
        voxel_grid = mesh.voxelized(pitch = pitch_val)
        voxel_data = voxel_grid.matrix
        padded_voxels = np.zeros((voxel_res, voxel_res, voxel_res), dtype=bool)
        offsets = [(voxel_res - min(s, voxel_res)) // 2 for s in voxel_data.shape]
        insert_slices = tuple(slice(offset, offset + min(s, voxel_res)) for offset, s in zip(offsets, voxel_data.shape))
        padded_voxels[insert_slices] = voxel_data[:voxel_res, :voxel_res, :voxel_res]
        mesh.visual = mesh.visual.to_color()
        if mesh.visual.kind == 'vertex':
            vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB values (ignore alpha if present)
        voxel_colors = np.zeros((voxel_res, voxel_res, voxel_res, 3), dtype=np.uint8)  # Initialize color grid
        for x in range(voxel_res):
            for y in range(voxel_res):
                for z in range(voxel_res):
                    if padded_voxels[x, y, z]:
                        voxel_position = np.array([x, y, z]) * pitch_val + mesh.bounds[0]
                        distances = np.linalg.norm(mesh.vertices - voxel_position, axis=1)
                        closest_vertex_idx = np.argmin(distances) # Find the nearest vertex from the original mesh
                        voxel_colors[x, y, z] = vertex_colors[closest_vertex_idx]
        visualize_voxel(voxel_colors)
        print('Voxel dimesnsions:', voxel_colors.shape)
        file_name = '_'.join(path.split('/'))
        np.save(f'{output_dir}/{file_name}', voxel_colors)

def main():
    create_voxel_grid()

if __name__ == '__main__':
    main()
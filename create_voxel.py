import trimesh
import numpy as np
import config
from ShapeNetCore import get_random_models
import config
import os
import matplotlib.pyplot as plt

def visualize_voxel(voxel_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_data, edgecolor='k')
    plt.show()

def create_voxel_grid():
    voxel_resolution = config.voxel_resolution
    output_dir = './voxel_data'
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(os.listdir(f'./triplane_images_{config.triplane_resolution}')):
        path = '/'.join(path.split('.')[0].split('_'))
        mesh_path = f'{config.pwd}/{path}/{config.suffix_dir}'
        mesh = trimesh.load(mesh_path, force='mesh')
        pitch_val = mesh.extents.max() / (voxel_resolution - 1)
        voxel_grid = mesh.voxelized(pitch = pitch_val)
        voxel_data = voxel_grid.matrix
        padded_voxels = np.zeros((64, 64, 64), dtype=bool)
        offsets = [(64 - min(s, 64)) // 2 for s in voxel_data.shape]
        insert_slices = tuple(slice(offset, offset + min(s, 64)) for offset, s in zip(offsets, voxel_data.shape))
        padded_voxels[insert_slices] = voxel_data[:64, :64, :64]
        print('Voxel dimesnsions:', padded_voxels.shape)
        # visualize_voxel(padded_voxels)
        file_name = '_'.join(path.split('/'))
        np.save(f'{output_dir}/{file_name}', padded_voxels)

def main():
    create_voxel_grid()

if __name__ == '__main__':
    main()
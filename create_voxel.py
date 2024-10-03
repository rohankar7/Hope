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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def create_voxel_grid():
    voxel_res = config.voxel_resolution
    output_dir = f'./voxel_data_{voxel_res}'
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(os.listdir(config.triplane_dir)[:50]):
        path = '/'.join(path.split('.')[0].split('_'))
        mesh_path = f'{config.pwd}/{path}/{config.suffix_dir}'
        mesh = trimesh.load(mesh_path, force='mesh')
        pitch_val = mesh.extents.max() / (voxel_res - 1)
        voxel_grid = mesh.voxelized(pitch = pitch_val)
        voxel_data = voxel_grid.matrix
        padded_voxels = np.zeros((voxel_res, voxel_res, voxel_res), dtype=bool)
        offsets = [(voxel_res - min(s, voxel_res)) // 2 for s in voxel_data.shape]
        insert_slices = tuple(slice(offset, offset + min(s, voxel_res)) for offset, s in zip(offsets, voxel_data.shape))
        padded_voxels[insert_slices] = voxel_data[:voxel_res, :voxel_res, :voxel_res]
        # offset = [(voxel_resolution - s) // 2 for s in voxel_data.shape]
        # padded_voxels[
        #     offset[0]:offset[0] + voxel_data.shape[0],
        #     offset[1]:offset[1] + voxel_data.shape[1],
        #     offset[2]:offset[2] + voxel_data.shape[2],
        # ] = voxel_data
        print('Voxel dimesnsions:', padded_voxels.shape)
        # visualize_voxel(padded_voxels)
        file_name = '_'.join(path.split('/'))
        np.save(f'{output_dir}/{file_name}', padded_voxels)

def main():
    create_voxel_grid()

if __name__ == '__main__':
    main()
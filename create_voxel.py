import trimesh
import numpy as np
import config
from ShapeNetCore import get_random_models
import config
import os

def create_voxel_grid():
    voxel_resolution = config.voxel_resolution
    output_dir = './voxel_data'
    os.makedirs(output_dir, exist_ok=True)
    os
    for path in sorted(get_random_models()):
        mesh_path = f'{config.pwd}/{path}/{config.suffix_dir}'
        mesh = trimesh.load(mesh_path, force='mesh')
        pitch_val = mesh.extents.max() / (voxel_resolution - 1)
        voxel_grid = mesh.voxelized(pitch = pitch_val)
        voxel_data = voxel_grid.matrix
        padded_voxels = np.zeros((64, 64, 64), dtype=bool)
        offset = [(64 - s) // 2 for s in voxel_data.shape]
        padded_voxels[
            offset[0]:offset[0] + voxel_data.shape[0],
            offset[1]:offset[1] + voxel_data.shape[1],
            offset[2]:offset[2] + voxel_data.shape[2],
        ] = voxel_data
        # voxel_dim = np.array(padded_voxels)
        print('Voxel dimesnsions:', padded_voxels.shape)
        file_name = '_'.join(path.split('/'))
        np.save(f'{output_dir}/{file_name}', padded_voxels)

def main():
    create_voxel_grid()

if __name__ == '__main__':
    main()
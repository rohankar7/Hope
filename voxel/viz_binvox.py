import binvox_rw
import numpy as np
import matplotlib.pyplot as plt

def visualize_voxel(voxel_data, threshold=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # if voxels.max() > 1:
    #     voxels = voxels / 255.0 # Normalizing the voxel colors for visualization
    # mask = np.any(voxels > threshold, axis=-1) # Masking for non-zero voxels with color intensity > 0
    # x, y, z = np.indices(voxels.shape[:-1])  # Getting the grid coordinates
    # ax.scatter(x[mask], y[mask], z[mask], c=voxels[mask].reshape(-1, 3), marker='o', s=20)
    # ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.voxels(voxel_data, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

import numpy as np
from binvox_rw import read_as_3d_array

def read_binvox_to_numpy(filename):
    # Load the .binvox file
    with open(filename, 'rb') as f:
        vox = read_as_3d_array(f)
    # Get the voxel data as a numpy array
    voxel_data = vox.data  # This will be a boolean array (depth, height, width)
    # Optionally, convert to an integer array (0 for empty voxels, 1 for filled voxels)
    voxel_data_int = voxel_data.astype(np.int_)
    return voxel_data_int

# data_dir = 'C:/ShapeNetCore/02691156/1943ee06ecb139819330265a9fff38de/models/model_normalized.solid.binvox'
# numpy_array = read_binvox_to_numpy(data_dir)
# visualize_voxel(numpy_array)

def save_numpy_to_binvox(voxel_data, filename, scale=1, translate=(0, 0, 0), axis_order='xyz'):
    """
    Save a NumPy array as a .binvox file.
    
    Parameters:
    - voxel_data: 3D NumPy array with 1 for filled voxels and 0 for empty.
    - filename: The path to save the .binvox file.
    - scale: Scale for the voxel model (default is 1).
    - translate: Translation for the voxel model as a tuple (default is (0, 0, 0)).
    - axis_order: The axis order, typically 'xyz'.
    """
    # Ensure voxel_data is a boolean array
    # voxel_data = (voxel_data > 0).astype(np.bool_)
    voxels = binvox_rw.Voxels(
        data=voxel_data,
        dims=voxel_data.shape,
        translate=translate,
        scale=scale,
        axis_order=axis_order
    )
    with open(filename, 'wb') as f:
        voxels.write(f)
dir_path = './voxel_data_64/02691156_3ae96a1e1bb488942296d88107d065f6.npy'
voxel_data = np.load(dir_path)
# voxel_data = voxel_data.astype(np.int_)
save_numpy_to_binvox(voxel_data, 'output.binvox')
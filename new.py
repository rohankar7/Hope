import trimesh
import numpy as np
import trimesh.voxel as voxel

# Parameters
input_obj_path = 'C:/ShapeNetCore/02691156/1a29042e20ab6f005e9e2656aff7dd5b/models/model_normalized.obj'
# input_obj_path = "C:/ShapeNetCore/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
output_ply_path = 'colored_voxel_mesh.ply'  # Output path for the colored voxel mesh
voxel_resolution = 64 # Resolution of the voxel grid

# Step 1: Load the 3D .obj file
mesh = trimesh.load(input_obj_path, force='mesh')
mesh.visual = mesh.visual.to_color()

# Step 2: Voxelize the Mesh
pitch = mesh.extents.max() / voxel_resolution
voxelized_mesh = mesh.voxelized(pitch=pitch)

# Step 3: Center the Voxel Grid to Match Mesh Bounds
# Calculate the voxel grid bounds
voxel_grid_min_bound = mesh.bounds[0] - (pitch / 2)
voxel_grid_max_bound = mesh.bounds[1] + (pitch / 2)

# Step 4: Map Colors to the Voxel Grid
# Initialize an array to store RGB colors for the voxel grid
voxel_colors = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution, 3), dtype=np.uint8)

# Iterate over each voxel position
for voxel_point in voxelized_mesh.points:
    # Convert the voxel point to the mesh's coordinate system
    x, y, z = ((voxel_point - voxel_grid_min_bound) / pitch).astype(int)

    # Check if the voxel is within bounds before proceeding
    if 0 <= x < voxel_resolution and 0 <= y < voxel_resolution and 0 <= z < voxel_resolution:
        # Find the nearest vertex to the current voxel point
        distances = np.linalg.norm(mesh.vertices - voxel_point, axis=1)
        nearest_vertex_index = np.argmin(distances)

        # Extract the nearest vertex's color (assuming colors are in mesh.visual.vertex_colors)
        color = mesh.visual.vertex_colors[nearest_vertex_index][:3]  # RGB values only
        voxel_colors[x, y, z] = color

print(voxel_colors.shape)
from create_voxel import visualize_voxel
visualize_voxel(voxel_colors)
# # Step 5: Convert Voxel Grid with Colors to Point Cloud Data for Export
# # Get the occupied voxel indices that have colors assigned
# occupied_voxel_indices = np.argwhere(np.any(voxel_colors != 0, axis=-1))

# # Convert voxel indices back to world coordinates
# points = occupied_voxel_indices * pitch + voxel_grid_min_bound
# colors = voxel_colors[occupied_voxel_indices[:, 0], occupied_voxel_indices[:, 1], occupied_voxel_indices[:, 2]]

# # Step 6: Export as a .ply file
# # Create a point cloud from the colored voxel points
# colored_voxel_mesh = trimesh.PointCloud(points, colors=colors)
# colored_voxel_mesh.show()
# # colored_voxel_mesh.export(output_ply_path)

# print(f"Colored voxel mesh saved to {output_ply_path}")
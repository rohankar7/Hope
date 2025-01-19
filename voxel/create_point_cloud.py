import trimesh
import os
import numpy as np
import open3d as o3d

path = 'C:/ShapeNetCore/02691156/1a29042e20ab6f005e9e2656aff7dd5b/models/model_normalized.obj'
mesh = o3d.io.read_triangle_mesh(path)

# Check if the mesh has vertex colors
if not mesh.has_vertex_colors():
    print("The mesh does not have vertex colors. Please ensure your mesh has color information.")
else:
    print("Loaded mesh with vertex colors.")

# Step 2: Sample points from the mesh
num_points = 50000  # Specify the number of points you want to sample
point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

# Extract vertex colors from the mesh
vertex_colors = np.asarray(mesh.vertex_colors)

# To assign colors to the point cloud, we will map sampled points to the closest vertices
# Compute the nearest vertex for each sampled point
pcd_points = np.asarray(point_cloud.points)
pcd_colors = []

# For each sampled point, find the nearest vertex and use its color
for point in pcd_points:
    # Get the index of the nearest vertex
    distances = np.linalg.norm(vertex_colors - point, axis=1)
    nearest_vertex_index = np.argmin(distances)
    pcd_colors.append(vertex_colors[nearest_vertex_index])

# Convert the list of colors to a numpy array
pcd_colors = np.array(pcd_colors)

# Step 3: Assign the colors to the point cloud
point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors)

# Visualize the colored point cloud
o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

# Optional: print some information about the point cloud
print(f"Sampled {len(pcd_points)} points.")
print(f"Colors shape: {pcd_colors.shape}")

# Save point cloud with colors as .ply if needed
# o3d.io.write_point_cloud("colored_point_cloud.ply", point_cloud)
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage.draw import polygon
from ShapeNetCore import *
import os
import config

triplane_resolution = config.triplane_resolution
dtype = np.float32

def compute_sdf(mesh, min_bound = -1, max_bound = 1, resolution=triplane_resolution):
    projections = ['xy', 'yz', 'zx']
    min_bound, max_bound = mesh.bounds  # Get the bounds of the mesh
    # Generate a grid of points
    x = np.linspace(min_bound[0], max_bound[0], num=resolution)
    y = np.linspace(min_bound[1], max_bound[1], num=resolution)
    z = np.linspace(min_bound[2], max_bound[2], num=resolution)
    xy_points, yz_points, zx_points = [], [], []
    for proj in projections:
        if proj == 'xy':
            for i in x:
                for j in y:
                    xy_points.append([i,j,min_bound[2]])
            xy_points = np.array(xy_points)
        elif proj == 'yz':
            for i in y:
                for j in z:
                    yz_points.append([min_bound[0],i,j])
            yz_points = np.array(yz_points)
        elif proj == 'zx':
            for i in z:
                for j in x:
                    zx_points.append([j,min_bound[1],i])
            zx_points = np.array(zx_points)
    sdf_xy = trimesh.proximity.signed_distance(mesh, xy_points).reshape((resolution, resolution))
    sdf_yz = trimesh.proximity.signed_distance(mesh, yz_points).reshape((resolution, resolution))
    sdf_zx = trimesh.proximity.signed_distance(mesh, zx_points).reshape((resolution, resolution))
    sdf_values = np.stack([sdf_xy, sdf_yz, sdf_zx], axis=0, dtype=dtype)
    return sdf_values

def project_to_plane(mesh, plane='xy', resolution=triplane_resolution):
    # Create a grid to store the projection and RGB colors
    grid = np.zeros((resolution, resolution, 3), dtype=dtype)
    count_grid = np.zeros((resolution, resolution, 1), dtype=dtype)  # To average colors
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    # Calculate the scaling factor to fit the mesh within the resolution
    scale = (resolution) / (max_bound - min_bound).max()
    # Scale the vertices to fit in the grid while preserving aspect ratios
    vertices = mesh.vertices.copy()
    vertices = (vertices - min_bound) * scale
    # Center the vertices in the grid
    center_offset = (resolution - (vertices.max(axis=0) + vertices.min(axis=0))) / 2
    if plane == 'xy':
        vertices[:, :2] += center_offset[:2]
    elif plane == 'yz':
        vertices[:, 1:] += center_offset[1:]
    elif plane == 'zx':
        vertices[:, [2, 0]] += center_offset[[2, 0]]

    # Ensure vertices are within the grid
    vertices = np.clip(vertices, 0, resolution - 1).astype(float)
    # Normalising colours
    vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize colours
    # face_colors = mesh.visual.face_colors[:, :3] / 255.0
    for face in mesh.faces:
        tri = vertices[face]
        color = np.mean(vertex_colors[face], axis=0, dtype=dtype)  # Average vertex colors for face color
        if plane == 'xy':
            # tri[:, 0] += triplane_resolution // 64
            points = tri[:, :2]
        elif plane == 'yz':
            points = tri[:, 1:]
        elif plane == 'zx':
            # tri[:, 0] += triplane_resolution // 32
            points = tri[:, [2, 0]]
        rr, cc = polygon(points[:, 1], points[:, 0], grid.shape[:2])
        grid[rr, cc] += color
        count_grid[rr, cc, 0] += 1  # Counting contributions
    count_grid[count_grid == 0] = 1 # Avoiding division by zero
    grid = (grid / count_grid).astype(dtype)  # Averaging the colors
    return grid

def viz_projections(xy_projection, yz_projection, zx_projection):
    # Visualizing projections
    cmap = 'viridis'   # Choosing 'gray' or 'viridis'
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('XY Projection')
    plt.imshow(xy_projection, cmap=cmap)
    plt.subplot(1, 3, 2)
    plt.title('YZ Projection')
    plt.imshow(yz_projection, cmap=cmap)
    plt.subplot(1, 3, 3)
    plt.title('ZX Projection')
    plt.imshow(zx_projection, cmap=cmap)
    plt.show() 

def generate_triplanes(file_path, resolution=triplane_resolution):
    mesh = trimesh.load(file_path, force='mesh')
    mesh.visual = mesh.visual.to_color()
    # Stacking the SDF values for each projection
    # sdf_grid = compute_sdf(mesh)
    # sdf_reshaped = sdf_grid[:, :, :, np.newaxis]
    # Generate binary projections
    xy_projection = project_to_plane(mesh, 'xy', resolution)
    yz_projection = project_to_plane(mesh, 'yz', resolution)
    zx_projection = project_to_plane(mesh, 'zx', resolution)
    zx_projection = np.rot90(zx_projection, k=-1)
    # viz_projections(xy_projection, yz_projection, zx_projection)    # Visualize the projection
    # Stacking the projections to create a triplane
    triplane = np.stack([xy_projection, yz_projection, zx_projection], axis=0)
    # triplane = np.concatenate((triplane, sdf_reshaped), axis=-1)
    return triplane

def model_to_triplanes(out_dir):
    for path in get_random_models():
        os.makedirs(out_dir, exist_ok=True)
        try:
            triplane = generate_triplanes(f'{pwd}/{path}/{suffix_dir}', resolution=triplane_resolution)
            # Triplane shape: 3 x N x N x 3
            np.save(f"{out_dir}/{'_'.join(path.split('/'))}.npy", triplane)
            print('Saved triplane')
        except IndexError as e:
            print('Skipped model:', path)
            continue

def main():
    print('Main function: Triplane')
    model_to_triplanes(f'./triplane_images_{triplane_resolution}')

if __name__ == "__main__":
    main()
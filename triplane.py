import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage.draw import polygon
from ShapeNetCore import *
import os

triplane_resolution = 256

def project_to_plane(mesh, plane='xy', resolution=triplane_resolution):
    # Create a grid to store the projection and RGB colors
    grid = np.zeros((resolution, resolution, 3), dtype=np.float32)
    count_grid = np.zeros((resolution, resolution, 1), dtype=np.int32)  # To average colors
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    # Calculate the scaling factor to fit the mesh within the resolution
    scale = (resolution - 1) / (max_bound - min_bound).max()
    # Normalize the vertices to fit in the grid
    vertices = mesh.vertices.copy()
    vertices = (vertices - min_bound) / (max_bound - min_bound)
    # vertices = np.clip(vertices * (resolution - 1), 0, resolution - 1).astype(int)
    # Scale the vertices to fit in the grid while preserving aspect ratios
    vertices = mesh.vertices.copy()
    vertices = (vertices - min_bound) * scale
    # vertices = np.clip(vertices, 0, resolution - 1).astype(int)
    # Normalising colours
    vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize colours
    face_colors = mesh.visual.face_colors[:, :3] / 255.0
    for face in mesh.faces:
        tri = vertices[face]
        color = np.mean(vertex_colors[face], axis=0)  # Average vertex colors for face color
        if plane == 'xy':
            points = tri[:, :2]
        elif plane == 'yz':
            points = tri[:, 1:]
        elif plane == 'zx':
            points = tri[:, [2, 0]]
        rr, cc = polygon(points[:, 1], points[:, 0], grid.shape[:2])
        grid[rr, cc] += color
        count_grid[rr, cc, 0] += 1  # Counting contributions
    
    # Avoiding division by zero
    count_grid[count_grid == 0] = 1
    grid /= count_grid  # Averaging the colors
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
    # mesh.apply_translation(-mesh.centroid)
    mesh.visual = mesh.visual.to_color()
    # Generate random colors
    if not mesh.visual.vertex_colors.any():
        mesh.visual.vertex_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=np.float32)
    if not mesh.visual.face_colors.any():
        mesh.visual.face_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=np.float32)

    # Generate binary projections
    xy_projection = project_to_plane(mesh, 'xy', resolution)
    yz_projection = project_to_plane(mesh, 'yz', resolution)
    zx_projection = project_to_plane(mesh, 'zx', resolution)
    zx_projection = np.rot90(zx_projection, k=-1)
    return xy_projection, yz_projection, zx_projection

def model_to_triplanes(out_dir, resolution=triplane_resolution):
    for path in subclasses_list[:5]:
        os.makedirs(out_dir, exist_ok=True)
        xy_projection, yz_projection, zx_projection_rotated = generate_triplanes(f'{pwd}/{path}/{suffix_dir}', resolution)
        triplane = np.stack([xy_projection, yz_projection, zx_projection_rotated], axis=0)
        print(triplane.shape)
        np.save(f"{out_dir}/{'_'.join(path.split('/'))}.npy", triplane)
        viz_projections(xy_projection, yz_projection, zx_projection_rotated)

def main():
    model_to_triplanes('./images', resolution=triplane_resolution)

if __name__ == "__main__":
    main()
import numpy as np
import trimesh
import torch
import os
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, disk
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
from ldm import *
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
triplane_res = config.triplane_resolution
voxel_res = config.voxel_resolution
# trimesh.exchange.export.export_mesh(mesh, triplane_fname[:-4] + '.ply', file_type='ply')

def clean_projection(projection, operation='closing', radius=2):
    selem = disk(radius)
    if operation == 'closing':
        return binary_closing(projection, selem)
    elif operation == 'opening':
        return binary_opening(projection, selem)
    else:
        raise ValueError("Invalid operation. Use 'closing' or 'opening'.")
    # return projection

def smooth_voxel_grid(voxel_grid, iterations=1):
    for _ in range(iterations):
        voxel_grid = binary_dilation(voxel_grid)
        voxel_grid = binary_erosion(voxel_grid)
    return voxel_grid

def fill_voxel_grid(xy_projection, yz_projection, zx_projection, resolution):
    voxel_grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.uint8)
    for x in range(resolution):
        for y in range(resolution):
            if xy_projection[y, x].any():  # If the pixel is part of the object in the XY plane
                for z in range(resolution):
                    if yz_projection[z, y].any() and zx_projection[z, x].any():
                        voxel_grid[x, y, z] = np.mean([xy_projection[y, x], yz_projection[z, y], zx_projection[z, x]], axis=0)
    return voxel_grid

def extract_colors(triplane, vertices, faces, resolution):
    xy_projection, yz_projection, zx_projection = triplane[..., :3]
    dtype = np.int64
    vertex_colors = np.zeros((vertices.shape[0], 3), dtype=dtype)
    face_colors = np.zeros((faces.shape[0], 3), dtype=dtype)
    # Extract color for the vertices
    for i, (x, y, z) in enumerate(vertices):
        x, y, z = int(x), int(y), int(z)
        # Get corresponding colors from each plane
        xy_color = xy_projection[y, x] if 0 <= x < resolution and 0 <= y < resolution else np.array([0, 0, 0])
        yz_color = yz_projection[z, y] if 0 <= y < resolution and 0 <= z < resolution else np.array([0, 0, 0])
        zx_color = zx_projection[z, x] if 0 <= z < resolution and 0 <= x < resolution else np.array([0, 0, 0])
        # Average the colors from the three planes
        vertex_colors[i] = np.mean([xy_color, yz_color, zx_color], axis=0) * 255
    # Extract face colors
    for i, face in enumerate(faces):
        face_colors[i] = np.mean(vertex_colors[face], axis=0) * 255
    return vertex_colors, face_colors

def generate_mesh(triplane, zx_projection, resolution=triplane_res):
    # xy_projection = triplane[0][:, :, 0] * triplane[0][:, :, 3]
    # yz_projection = triplane[1][:, :, 0] * triplane[1][:, :, 3]
    xy_projection = triplane[0][:, :, 0]
    yz_projection = triplane[1][:, :, 0]
    zx_projection_rotated = zx_projection
    # Clean the projections
    xy_projection = clean_projection(xy_projection, 'closing', 2)
    yz_projection = clean_projection(yz_projection, 'closing', 2)
    zx_projection_rotated = clean_projection(zx_projection_rotated, 'closing', 2)
    voxel_grid = fill_voxel_grid(xy_projection, yz_projection, zx_projection_rotated, resolution)
    voxel_grid_smoothed = smooth_voxel_grid(voxel_grid[..., 0], iterations=2)
    # Visualize the voxel grid
    threshold, min_value, max_value = None, voxel_grid_smoothed.min(), voxel_grid_smoothed.max()
    if threshold is None: threshold = (min_value + max_value) / 2
    if not (min_value <= threshold <= max_value): # Ensure the threshold is within the data range
        raise ValueError(f"Threshold {threshold} is out of range ({min_value}, {max_value})")
    if max_value == min_value:
        raise RuntimeError('The voxel grid is empty or uniform after smoothing.')
    verts, faces, normals, values = measure.marching_cubes(voxel_grid_smoothed, level=threshold)
    vertex_colors, face_colors = extract_colors(triplane, verts, faces, resolution)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, vertex_attributes={'values':values}, vertex_colors=vertex_colors, face_colors=face_colors)
    # mesh.show()
    return mesh

def repair_mesh(mesh):
    mesh.fill_holes()  # Fill holes in the mesh
    mesh.update_faces(mesh.nondegenerate_faces()) # Remove degenerate faces
    # mesh.update_faces(mesh.unique_faces())  # Remove duplicate faces
    mesh.remove_infinite_values()  # Remove infinite values
    mesh.remove_unreferenced_vertices()  # Remove unreferenced vertices
    mesh.update_faces(mesh.unique_faces())  # Remove duplicate vertices
    return mesh

def correct_rotation(triplane, resolution):
    # zx_projection_rotated = triplane[2][:, :, 0] * triplane[2][:, :, 3]
    zx_projection_rotated = triplane[2][:, :, 0]
    try:
        mesh = generate_mesh(triplane, zx_projection_rotated, resolution)
        print('Error 1')
        return repair_mesh(mesh)
    except RuntimeError as e:
        try:
            zx_projection_rotated = np.rot90(zx_projection_rotated, k=2)
            mesh = generate_mesh(triplane, zx_projection_rotated, resolution)
            print('Error 2')
            return repair_mesh(mesh)
        except RuntimeError as e:
            try:
                zx_projection_rotated = np.rot90(zx_projection_rotated, k=-1)
                mesh = generate_mesh(triplane, zx_projection_rotated, resolution)
                print('Error 3')
                return repair_mesh(mesh)
            except RuntimeError as e:
                print("No surface found at the given iso value")
    return

def model_from_triplanes(triplane_dir):
    for np_triplane in os.listdir(triplane_dir)[:10]:
        triplane = np.load(os.path.join(triplane_dir, np_triplane))
        mesh = correct_rotation(triplane, triplane_res)
        mesh.show()
        model_gen_dir = './generated_models'
        os.makedirs(model_gen_dir, exist_ok=True)
        # trimesh.exchange.export.export_mesh(mesh, f"{model_gen_dir}/{np_triplane.split('.')[0]}", file_type='ply')
        mesh.export(f"{model_gen_dir}/{np_triplane.split('.')[0]}.ply")
        print('Exported')

def main():
    model_from_triplanes(config.triplane_dir)

if __name__ == "__main__":
    main()
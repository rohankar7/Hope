import numpy as np
import trimesh
from skimage import measure
import mcubes
from create_voxel import visualize_voxel

def create_colored_voxels(input_obj_path, voxel_resolution=128):
    mesh = trimesh.load(input_obj_path, force='mesh')
    mesh.visual = mesh.visual.to_color()
    pitch = mesh.extents.max() / (voxel_resolution)
    voxelized_mesh = mesh.voxelized(pitch=pitch)
    colored_voxel_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution, 4), dtype=np.float32)
    mesh_center = mesh.bounds.mean(axis=0)
    voxel_center = voxelized_mesh.bounds.mean(axis=0)
    offset = mesh_center - voxel_center
    for voxel_point in voxelized_mesh.points:
        voxel_point += offset
        distances = np.linalg.norm(mesh.vertices - voxel_point, axis=1)
        nearest_vertex_index = np.argmin(distances)
        color = mesh.visual.vertex_colors[nearest_vertex_index][:3] / 255.0  # RGB values only
        x, y, z = np.round((voxel_point - voxelized_mesh.bounds[0]) / pitch).astype(int)
        if 0 <= x < voxel_resolution and 0 <= y < voxel_resolution and 0 <= z < voxel_resolution:
            colored_voxel_grid[x, y, z, :3] = color  # Assign RGB color
            colored_voxel_grid[x, y, z, 3] = 1     # Set occupancy flag (255 for occupied)
    np.save(f'{voxel_resolution}', colored_voxel_grid)

def get_interpolated_color(vertex_coord, voxel_grid):
    x, y, z = vertex_coord
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    # Ensure neighbors are within bounds
    x1 = min(x1, voxel_grid.shape[0] - 1)
    y1 = min(y1, voxel_grid.shape[1] - 1)
    z1 = min(z1, voxel_grid.shape[2] - 1)
    
    # Fetch colors of the eight neighbors
    c000 = voxel_grid[x0, y0, z0, :3] / 255.0
    c001 = voxel_grid[x0, y0, z1, :3] / 255.0
    c010 = voxel_grid[x0, y1, z0, :3] / 255.0
    c011 = voxel_grid[x0, y1, z1, :3] / 255.0
    c100 = voxel_grid[x1, y0, z0, :3] / 255.0
    c101 = voxel_grid[x1, y0, z1, :3] / 255.0
    c110 = voxel_grid[x1, y1, z0, :3] / 255.0
    c111 = voxel_grid[x1, y1, z1, :3] / 255.0
    
    # Calculate weights for each dimension based on the fractional part
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Interpolate along x
    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx
    
    # Interpolate along y
    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy
    
    # Interpolate along z for the final color
    interpolated_color = c0 * (1 - dz) + c1 * dz
    
    return interpolated_color

def create_mesh_from_colored_voxel_grid(colored_voxel_grid):
    # Step 1: Create a binary occupancy grid for the Marching Cubes algorithm
    binary_voxel_grid = (colored_voxel_grid[..., 3] > 0.5).astype(np.float32)  # Occupied if fourth channel > 0
    from generator import mesh_from_voxel
    # Step 2: Apply Marching Cubes to extract the surface mesh
    # mesh = mesh_from_voxel(binary_voxel_grid)
    verts, faces, normals, _ = measure.marching_cubes(binary_voxel_grid, level=0.51)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
    # Step 3: Map colors to vertices
    # Scale verts to fit the voxel space (if necessary)
    # verts = mesh.vertices

    # Initialize an array to store colors for each vertex
    vertex_colors = np.zeros((verts.shape[0], 3))
    grid_shape = colored_voxel_grid.shape[:3]
    # Find nearest voxel color for each vertex
    for i, vert in enumerate(verts):
        x, y, z = np.round(vert).astype(int)  # Get nearest voxel coordinate
        # # if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
        vertex_colors[i] = colored_voxel_grid[x, y, z, :3]  # Normalize RGB color to [0, 1]
    mesh.visual.vertex_colors = vertex_colors
    # Step 4: Create a Trimesh mesh with vertex colors
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, face_normals=normals)
    mesh.show()
    mesh.export('new.ply')
    return mesh # Returns the colored mesh

input_obj_path = 'C:/ShapeNetCore/02691156/1a29042e20ab6f005e9e2656aff7dd5b/models/model_normalized.obj'
# input_obj_path = "C:/ShapeNetCore/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
output_ply_path = 'colored_voxel_mesh.ply'  # Output path for the colored voxel mesh
voxel_resolution = 128 # Resolution of the voxel grid
# create_colored_voxels(input_obj_path)
colored_voxel = np.load(f'./{voxel_resolution}.npy')
# visualize_voxel(colored_voxel)
mesh = create_mesh_from_colored_voxel_grid(colored_voxel)
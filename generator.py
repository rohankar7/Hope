import numpy as np
import trimesh
import torch
from torch import optim
from openai import OpenAI
import os
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, disk
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
from ganesha import UNetWithCrossAttention
from vae import VAE
from ShapeNetCore import *
from triplane import triplane_resolution
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# Additional file formats
# mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=(rgb_final * 255).astype(np.uint8))
#     if save_name:
#         trimesh.exchange.export.export_mesh(mesh, save_name, file_type='ply')
#     else:
#         trimesh.exchange.export.export_mesh(mesh, triplane_fname[:-4] + '.ply', file_type='ply')

def load_ldm_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

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

def generate_mesh(triplane, zx_projection, resolution=triplane_resolution):
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
    mesh.show()
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
        return mesh
    except RuntimeError as e:
        try:
            zx_projection_rotated = np.rot90(zx_projection_rotated, k=2)
            mesh = generate_mesh(triplane, zx_projection_rotated, resolution)
            print('Error 2')
            return mesh
        except RuntimeError as e:
            try:
                zx_projection_rotated = np.rot90(zx_projection_rotated, k=-1)
                mesh = generate_mesh(triplane, zx_projection_rotated, resolution)
                print('Error 3')
                return mesh
            except RuntimeError as e:
                print("No surface found at the given iso value")
    return

def model_from_triplanes(output_dir, resolution):
    for np_triplane in os.listdir(output_dir):
        triplane = np.load(os.path.join(output_dir, np_triplane))
        mesh = correct_rotation(triplane, resolution)
        mesh = repair_mesh(mesh) # Mesh repairs
        model_gen_dir = './generated_models'
        os.makedirs(model_gen_dir, exist_ok=True)
        mesh.export(f"{model_gen_dir}/{np_triplane.split('.')[0]}.ply")
        print('Exported')

def generate_from_text(text, num_steps=1000):
    ldm = UNetWithCrossAttention().to(device)
    optimizer = optim.Adam(ldm.parameters(), lr=1e-4)
    checkpoint_path = './ldm_checkpoints/ldm_epoch_0.pth'
    ldm, optimizer, start_epoch = load_ldm_checkpoint(ldm, optimizer, checkpoint_path)
    ldm.to(device)
    ldm.eval()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    embedding_model = "text-embedding-3-small"
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input = [text], model=embedding_model).data[0].embedding
    embedding = torch.tensor(embedding).to(device)
    # data_size = (1, 12, 32, 32)  # Size of the latent data
    data_size = (1, 9, 256, 256)
    current = torch.randn(data_size).to(device)  # Starting with random noise
    # Reverse diffusion process
    for step in range(num_steps - 1, -1, -1):
        time_frac = step / float(num_steps)
        noise_level = np.cos((1.0 - time_frac) * np.pi / 2) ** 2  # Cosine noise schedule
        # Model predicts the reverse of the noise
        predicted_noise = ldm(current, embedding)
        current = current - predicted_noise * noise_level  # Reverse the noise addition
    print('Current', current.shape)
    return current

def decode_latent_triplanes(latent_triplanes):
    latent_dim = 64
    vae  = VAE().to(device)
    vae.load_state_dict(torch.load('./vae_weights/weights.pth'))
    vae.eval()
    latent_triplanes = torch.load('./latents/latent_0.pt').to(device)
    # latent_triplanes = ldm(latent_vectors.unsqueeze(0))
    # latent_triplanes = ldm(latent_vectors)
    decoded_triplanes = vae.decode(latent_triplanes[:, :3, :, :]).permute(0, 3, 2, 1).contiguous()
    return decoded_triplanes

def main():
    text = 'A white aeroplane with red wings'
    coarse_latent_data = generate_from_text(text)
    # coarse_triplanes = decode_latent_triplanes(coarse_latent_data).cpu().detach().numpy()
    coarse_triplanes = coarse_latent_data.cpu().detach().numpy()
    triplane_savedir = './generated_triplanes'
    os.makedirs(triplane_savedir, exist_ok=True)
    np.save(f"{triplane_savedir}/{'output'}.npy", coarse_triplanes)
    # triplane_savedir = './images'
    model_from_triplanes(triplane_savedir, resolution=triplane_resolution)

if __name__ == "__main__":
    main()
import numpy as np
import trimesh
import torch
from torch import optim
from openai import OpenAI
import os
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, disk
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
from ldm import UNetWithCrossAttention
from vae import VAE
from ldm import *
import math
import config
from mlp import TriplaneMLP
from create_voxel import visualize_voxel
# from skimage import measure
import mcubes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
triplane_res = config.triplane_resolution
voxel_res = config.voxel_resolution
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

def repair_mesh(mesh):
    mesh.fill_holes()  # Fill holes in the mesh
    mesh.update_faces(mesh.nondegenerate_faces()) # Remove degenerate faces
    # mesh.update_faces(mesh.unique_faces())  # Remove duplicate faces
    mesh.remove_infinite_values()  # Remove infinite values
    mesh.remove_unreferenced_vertices()  # Remove unreferenced vertices
    mesh.update_faces(mesh.unique_faces())  # Remove duplicate vertices
    return mesh

# 2, 5
def create_mesh_from_voxel(mlp_voxel):
    threshold = 0.5
    # visualize_voxel(mlp_voxel.squeeze().numpy(), threshold=threshold)
    visualize_voxel(mlp_voxel, threshold=threshold)
    # mlp_voxel = torch.sum(mlp_voxel, axis=3)
    # voxel_grid_binary = (mlp_voxel > threshold).int()
    # voxel_grid_np = voxel_grid_binary.squeeze().numpy()
    # mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid_np)
    # vertex_indices = mesh.vertices.astype(int)
    # vertex_indices = np.clip(vertex_indices, 0, np.array(voxel_grid_np.shape[:3]) - 1)
    # vertex_colors = voxel_grid_np[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2], :]
    # if vertex_colors.max() <= 1.0:
    #     vertex_colors = vertex_colors * 255.0
    # mesh.visual.vertex_colors = vertex_colors
    # return mesh
    # from scipy.ndimage import gaussian_filter
    # # mlp_voxel = mlp_voxel.cpu().numpy()
    # # mlp_voxel = gaussian_filter(mlp_voxel, sigma=1)
    # vertices, faces, normals, values = measure.marching_cubes(mlp_voxel)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, face_normals=values)
    # mesh.show()
    # return mesh
    # vertices, triangles = mcubes.marching_cubes(mlp_voxel, 0.5)
    # vertices = vertices / (256 - 1.0) * 2 - 1
    # mesh = trimesh.Trimesh(vertices, triangles)
    # return mesh
    # voxel_grid_np = mlp_voxel.squeeze().numpy()  # Assuming shape (N, N, N, 3)
    # import scipy
    # upscale_factor = 2
    voxel_grid_np = mlp_voxel
    # voxel_grid_grayscale = np.max(voxel_grid_np, axis=3)
    voxel_grid_binary = (voxel_grid_np > threshold).astype(int)
    # voxel_grid_np = scipy.ndimage.zoom(voxel_grid_binary, upscale_factor, order=1)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid_binary)
    # subdivided_mesh = mesh.subdivide()

    # If needed, repeat subdivision multiple times to increase smoothness
    # for _ in range(10):  # Apply subdivision multiple times for smoother results
        # subdivided_mesh = subdivided_mesh.subdivide()

    mesh.show()
    return mesh
    # voxel_object = trimesh.voxel.VoxelGrid(voxel_grid_binary)
    # mesh = voxel_object.marching_cubes
    vertex_indices = mesh.vertices.astype(np.uint8)
    vertex_indices = np.clip(vertex_indices, 0, np.array(voxel_grid_np.shape[:3]) - 1)
    vertex_colors = voxel_grid_np[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2], :]
    if vertex_colors.max() > 1.0:
        vertex_colors = vertex_colors / 255.0
    mesh.visual.vertex_colors = vertex_colors
    face_centroids = mesh.triangles_center.astype(int)

    # Ensure indices are within the voxel grid bounds
    face_centroids = np.clip(face_centroids, 0, np.array(voxel_grid_np.shape[:3]) - 1)

    # Extract the RGB values for each face based on the centroids
    face_colors = voxel_grid_np[face_centroids[:, 0], face_centroids[:, 1], face_centroids[:, 2], :]

    # Normalize the colors to [0, 255] for trimesh compatibility if they are in [0, 1]
    if face_colors.max() <= 1.0:
        face_colors = (face_colors * 255).astype(np.uint8)

    # Add alpha channel to make RGBA colors
    alpha_channel = np.full((face_colors.shape[0], 1), 255, dtype=np.uint8)  # Alpha value of 255 (fully opaque)
    face_colors = np.hstack([face_colors, alpha_channel])  # Combine RGB with Alpha to form RGBA

    # Assign the face colors to the mesh
    mesh.visual.face_colors = face_colors

    return mesh

def mesh_from_mlp(triplane):
    triplane_in_dim = 3 * triplane_res * triplane_res * config.triplane_features
    voxel_out_dim = voxel_res * voxel_res * voxel_res * 3
    model = TriplaneMLP(triplane_in_dim, voxel_out_dim)
    model.load_state_dict(torch.load('./mlp_weights/mlp_weights_80.pth'))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(triplane.reshape(triplane_in_dim), dtype=torch.float32)
        output = model(input_tensor)
        mesh = create_mesh_from_voxel(output)
        mesh = repair_mesh(mesh)
        # mesh = smooth_voxel_grid(mesh)
        mesh.show()
    return mesh

def model_from_triplanes(output_dir):
    for np_triplane in os.listdir(output_dir)[:10]:
        triplane = np.load(os.path.join(output_dir, np_triplane))
        mesh = mesh_from_mlp(triplane)
        model_gen_dir = './generated_models'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_gen_dir, exist_ok=True)
        mesh.export(f"{model_gen_dir}/{np_triplane.split('.')[0]}.ply")
        print('Exported')

def generate_from_text(text):
    timesteps = 1000
    ldm = UNetWithCrossAttention().to(device)
    optimizer = optim.Adam(ldm.parameters(), lr=1e-4)
    checkpoint_path = './ldm_checkpoints/ldm_epoch_8.pth'
    ldm, optimizer, start_epoch = load_ldm_checkpoint(ldm, optimizer, checkpoint_path)
    ldm.to(device)
    ldm.eval()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    embedding_model = "text-embedding-3-small"
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input = [text], model=embedding_model).data[0].embedding
    embedding = torch.tensor(embedding).to(device)
    # data_size = (1, 12, 32, 32)  # Size of the latent data
    data_size = (3, 3, 128, 128)
    noise_scheduler = NoiseScheduler(timesteps, linear_beta_schedule)
    x_t = torch.randn(data_size).to(device)  # Starting with random noise
    # Reverse diffusion process
    for t in reversed(range(noise_scheduler.timesteps)):
        predicted_noise = ldm(x_t, t)
        x_t = noise_scheduler.predict_start_from_noise(x_t, t, predicted_noise)
    return x_t

def decode_latent_triplanes(latent_triplanes):
    latent_dim = 64
    vae  = VAE().to(device)
    vae.load_state_dict(torch.load('./vae_weights/weights.pth'))
    vae.eval()
    latent_triplanes = torch.load('./latents/latent_0.pt').to(device)
    decoded_triplanes = vae.decode(latent_triplanes[:, :3, :, :]).permute(0, 3, 2, 1).contiguous()
    return decoded_triplanes

def main():
    triplane_savedir = './generated_triplanes'
    triplane_savedir = f'./triplane_images_{triplane_res}'
    # text = 'A white aeroplane with red wings'
    # coarse_latent_data = generate_from_text(text)
    # # print(coarse_latent_data.shape)
    # coarse_triplanes = decode_latent_triplanes(coarse_latent_data).cpu().detach().numpy()
    # coarse_triplanes = coarse_latent_data.cpu().detach().numpy()
    # np.save(f"{triplane_savedir}/{'output'}.npy", coarse_triplanes)

    # triplane_savedir = './triplane_images_128'
    model_from_triplanes(triplane_savedir)

if __name__ == "__main__":
    main()
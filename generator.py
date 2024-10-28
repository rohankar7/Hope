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

def load_ldm_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

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
def colored_mesh_from_voxel(mlp_voxel):
    threshold = 0
    if isinstance(mlp_voxel, np.ndarray):
        mlp_voxel = torch.from_numpy(mlp_voxel)
    voxel_grid_binary = (torch.sum(mlp_voxel, axis=3) > threshold).int()
    voxel_grid_np = voxel_grid_binary.numpy()
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid_np)
    vertex_indices = mesh.vertices.astype(int)
    vertex_indices = np.clip(vertex_indices, 0, np.array(voxel_grid_np.shape) - 1)
    mlp_voxel_np = mlp_voxel.numpy()
    vertex_colors = mlp_voxel_np[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2], :]
    if vertex_colors.max() <= 1.0:
        vertex_colors = (vertex_colors * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = vertex_colors
    mesh.show()
    return mesh

def mesh_from_voxel(mlp_voxel):
    if isinstance(mlp_voxel, np.ndarray):
        mlp_voxel = torch.from_numpy(mlp_voxel)
    threshold = 0.5
    voxel_grid_binary = (mlp_voxel > threshold).int()
    voxel_grid_np = voxel_grid_binary.squeeze().numpy()
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid_np)
    return mesh

def mesh_from_mlp(triplane):
    triplane_in_dim = config.triplane_planes * (config.triplane_resolution ** 2) * config.triplane_features
    model = TriplaneMLP()
    model.load_state_dict(torch.load('./mlp_weights/mlp_weights_300.pth'))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(triplane.reshape(triplane_in_dim), dtype=torch.float32)
        output = model(input_tensor)
        mesh = mesh_from_voxel(output)
        mesh = repair_mesh(mesh)
        # mesh = smooth_voxel_grid(mesh)
        mesh.show()
    return mesh

def model_from_triplanes(output_dir):
    for np_triplane in sorted(os.listdir(output_dir)[:10]):
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
    data_size = (3, 3, config.triplane_resolution, config.triplane_resolution)
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
    triplane_savedir = f'./triplane_images_{config.triplane_resolution}_alpha'
    # text = 'A white aeroplane with red wings'
    # coarse_latent_data = generate_from_text(text)
    # # print(coarse_latent_data.shape)
    # coarse_triplanes = decode_latent_triplanes(coarse_latent_data).cpu().detach().numpy()
    # coarse_triplanes = coarse_latent_data.cpu().detach().numpy()
    # np.save(f"{triplane_savedir}/{'output'}.npy", coarse_triplanes)

    # triplane_savedir = f'./triplane_images_{triplane_res}'
    model_from_triplanes(triplane_savedir)
    # mesh_from_np_voxel()

if __name__ == "__main__":
    main()
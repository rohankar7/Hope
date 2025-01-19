import os
from ShapeNetCore import *
# import trimesh
# import numpy as np
# from Model_List import model_paths
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # file_path = 'C:/ShapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj'
# # mesh = trimesh.load('./generated_models/02747177_10839d0dc35c94fcf4fb4dee5181bee_rotated.ply', force='mesh')
# # mesh.show()


# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, embedding_dim, num_heads=1):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.scale = (feature_dim // num_heads) ** -0.5
#         # Query, Key, Value projections for image features
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(embedding_dim, feature_dim)
#         self.value = nn.Linear(embedding_dim, feature_dim)
#         # Output projection layer
#         self.proj_out = nn.Linear(feature_dim, feature_dim)
        
#     def forward(self, img_features, text_embeddings):
#         # img_features shape: (batch_size, num_img_tokens, feature_dim)
#         # text_embeddings shape: (batch_size, num_text_tokens, embedding_dim)
#         # Project image features and text embeddings
#         query = self.query(img_features)  # (batch_size, num_img_tokens, feature_dim)
#         key = self.key(text_embeddings)  # (batch_size, num_text_tokens, feature_dim)
#         value = self.value(text_embeddings)  # (batch_size, num_text_tokens, feature_dim)
#         # Reshape for multi-head attention
#         def split_heads(x):
#             # Split the last dimension into (heads, depth)
#             new_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
#             return x.view(*new_shape).permute(0, 2, 1, 3)  # (batch_size, num_heads, tokens, depth)
#         query = split_heads(query)
#         key = split_heads(key)
#         value = split_heads(value)
#         # Scaled dot-product attention
#         scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
#         attention = F.softmax(scores, dim=-1)
#         context = torch.matmul(attention, value)
#         # Concatenate heads and project
#         context = context.permute(0, 2, 1, 3).contiguous()
#         context_shape = context.size()[:-2] + (context.size(-2) * context.size(-1),)
#         context = context.view(*context_shape)
#         return self.proj_out(context)

# class UNetWithCrossAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # num_feature_channels = 3 * 4
#         num_feature_channels = 3 * 3
#         # Define the standard UNet layers
#         self.enc1 = nn.Conv2d(num_feature_channels, 64, kernel_size=3, padding=1)
#         self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
#         # Cross attention layer
#         self.cross_attention = CrossAttention(256, 1536)
        
#         # Continue with the rest of the UNet
#         self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec3 = nn.Conv2d(64, num_feature_channels, kernel_size=3, padding=1)

#     def forward(self, x, embedding):
#         x1 = F.relu(self.enc1(x))
#         x2 = F.relu(self.enc2(x1))
#         x3 = F.relu(self.enc3(x2))
        
#         # Apply cross-attention
#         x3 = self.cross_attention(x3, embedding)
        
#         # Decoding
#         x = F.relu(self.dec1(x3))
#         x = F.relu(self.dec2(x))
#         x = self.dec3(x)
#         return x


# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, embed_dim, num_heads=1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.query_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
#         self.key_conv = nn.Linear(embed_dim, feature_dim)
#         self.value_conv = nn.Linear(embed_dim, feature_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

#     def forward(self, x, embedding):
#         # x: feature maps from the UNet [batch_size, feature_dim, H, W]
#         # embedding: text embeddings [batch_size, embed_dim]
#         print(x.shape)
#         batch_size, feature_dim, H, W = x.shape
#         # print(batch_size, feature_dim, H, W)
#         # Prepare queries from feature maps
#         query = self.query_conv(x)
#         query = query.view(batch_size, feature_dim, -1).permute(2, 0, 1).contiguous()  # [HW, batch_size, feature_dim]
#         # Prepare keys and values from embeddings
#         key = self.key_conv(embedding)
#         value = self.value_conv(embedding)
#         # Expanding keys and values
#         key = key.unsqueeze(0).expand(H * W, -1, -1)  # Repeat keys for each spatial location
#         value = value.unsqueeze(0).expand(H * W, -1, -1)  # Repeat values for each spatial location
#         # Compute attention
#         attended, _ = self.attention(query, key, value)
#         attended = attended.permute(1, 2, 0).contiguous().view(batch_size, feature_dim, H, W)
#         # Combine attended features and input features
#         combined_features = x + attended
#         return combined_features



# ________________________________________________________________


# class CrossAttention(nn.Module):
#     def __init__(self, features_dim, embed_dim, num_heads=8):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.features_dim = features_dim
#         self.embed_dim = embed_dim
#         self.scale = (self.features_dim // num_heads) ** -0.5

#         self.to_q = nn.Linear(features_dim, features_dim, bias=False)
#         self.to_kv = nn.Linear(embed_dim, features_dim * 2, bias=False)
#         # self.to_out = nn.Linear(features_dim, features_dim)

#     def forward(self, x, embedding):
#         b, _, h, w = x.shape

#         # Query from feature maps
#         q = self.to_q(x.flatten(2).transpose(1, 2))  # Shape: (batch_size, height*width, features_dim)
#         q = q.view(b, h * w, self.num_heads, self.features_dim // self.num_heads).permute(0, 2, 1, 3)

#         # Key and value from embedding vector
#         kv = self.to_kv(embedding.expand(b, -1)).view(b, 2, self.num_heads, self.features_dim // self.num_heads).permute(1, 2, 0, 3)
#         k, v = kv[0], kv[1]

#         # Scaled Dot-Product Attention
#         q = q * self.scale
#         attn = torch.matmul(q, k.transpose(-2, -1))
#         attn = F.softmax(attn, dim=-1)

#         # Aggregate values
#         out = torch.matmul(attn, v)
#         out = out.transpose(1, 2).contiguous().view(b, h * w, self.features_dim)
#         out = out.view(b, self.features_dim, h, w)  # Reshape to (batch_size, features_dim, height, width)
#         # Final linear transformation
#         # out = self.to_out(out)

#         return out


# # Example of using the modified CrossAttention module
# batch_size = 3
# features_dim = 256
# height = 64
# width = 64
# embed_dim = 1536

# # Input feature maps and single batch embedding
# feature_maps = torch.randn(batch_size, features_dim, height, width)
# embedding = torch.randn(1, embed_dim)  # Single embedding for all batches

# # Create the CrossAttention layer
# cross_attention = CrossAttention(features_dim, embed_dim)

# # Forward pass
# output = cross_attention(feature_maps, embedding)  # Use the single embedding across all batches
# print(output.shape)  # Expected output: (3, 256, 64, 64)

# from triplane import viz_projections
# d = './triplanes_256_alpha'
# for triplanes in os.listdir(d):
#     t = np.load(os.path.join(d, triplanes))
#     viz_projections(t)
# import trimesh
# import torch
# def mesh_from_np_voxel(mlp_voxel):
#     threshold = 0
#     if isinstance(mlp_voxel, np.ndarray):
#         mlp_voxel = torch.from_numpy(mlp_voxel)  # Shape (72, 72, 72, 3)
#     voxel_grid_binary = (torch.sum(mlp_voxel, axis=3) > threshold).int()
#     voxel_grid_np = voxel_grid_binary.numpy()  # Shape (72, 72, 72)
#     mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid_np)
#     vertex_indices = mesh.vertices.astype(int)
#     vertex_indices = np.clip(vertex_indices, 0, np.array(voxel_grid_np.shape) - 1)
#     mlp_voxel_np = mlp_voxel.numpy()  # Convert mlp_voxel to numpy with shape (72, 72, 72, 3)
#     vertex_colors = mlp_voxel_np[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2], :]
#     if vertex_colors.max() <= 1.0:
#         vertex_colors = (vertex_colors * 255.0).astype(np.uint8)
#     mesh.visual.vertex_colors = vertex_colors
#     mesh.show()
#     return mesh

# from create_voxel import visualize_voxel
# for data in os.listdir('./colored_voxels')[:10]:
#     mlp_voxel = np.load(os.path.join(f'./colored_voxels', data))
#     mesh_from_np_voxel(mlp_voxel)
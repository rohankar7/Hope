import os
from ShapeNetCore import *
import trimesh
import numpy as np
# file_path = 'C:/ShapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj'
# mesh = trimesh.load('./generated_models/02747177_10839d0dc35c94fcf4fb4dee5181bee_rotated.ply', force='mesh')
# mesh.show()


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
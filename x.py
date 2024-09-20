import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, channel_dim, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.channel_dim = channel_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_dim = 64  # Query dimensions after projection
        
        # Linear transformations for query, key, and value
        self.query_proj = nn.Linear(channel_dim, self.query_dim * num_heads)
        self.key_proj = nn.Linear(embed_dim, self.query_dim * num_heads)
        self.value_proj = nn.Linear(embed_dim, self.query_dim * num_heads)

        # Output projection layer
        self.output_proj = nn.Linear(self.query_dim * num_heads, channel_dim)

        # Scaling factor to normalize the dot products
        self.scale = 1. / (self.query_dim ** 0.5)

    def forward(self, x, embedding):
        batch_size, _, height, width = x.shape
        
        # Flatten the spatial dimensions for processing
        x_flat = x.view(batch_size, 3, -1).permute(0, 2, 1)  # Shape: [batch, height*width, channels]

        # Expand the embedding to match the batch size
        embedding = embedding.repeat(batch_size, 1)

        # Project queries from the image and keys/values from the embedding
        q = self.query_proj(x_flat).view(batch_size, -1, self.num_heads, self.query_dim).permute(0, 2, 1, 3)
        k = self.key_proj(embedding).view(batch_size, self.num_heads, 1, self.query_dim).permute(0, 1, 3, 2)
        v = self.value_proj(embedding).view(batch_size, self.num_heads, 1, self.query_dim)
        
        # Attention mechanism
        attn_scores = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.channel_dim)

        # Project the output back to channel dimension and reshape to original image dimensions
        output = self.output_proj(context).view(batch_size, self.channel_dim, height, width)

        return output

# Example usage
batch_size = 3
img = torch.rand(batch_size, 3, 64, 64)  # Example image batch
embedding = torch.rand(1, 1536)  # Example single shared embedding

cross_attn = CrossAttention(channel_dim=3, embed_dim=1536, num_heads=8)
output = cross_attn(img, embedding)
print(output.shape)  # Should match the input image shape (3, 3, 64, 64)
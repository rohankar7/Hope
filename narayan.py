import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, features_dim, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.scale = (self.features_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(features_dim, features_dim, bias=False)
        self.to_kv = nn.Linear(embed_dim, features_dim * 2, bias=False)
        # self.to_out = nn.Linear(features_dim, features_dim)

    def forward(self, x, embedding):
        b, _, h, w = x.shape

        # Query from feature maps
        q = self.to_q(x.flatten(2).transpose(1, 2))  # Shape: (batch_size, height*width, features_dim)
        q = q.view(b, h * w, self.num_heads, self.features_dim // self.num_heads).permute(0, 2, 1, 3)

        # Key and value from embedding vector
        kv = self.to_kv(embedding.expand(b, -1)).view(b, 2, self.num_heads, self.features_dim // self.num_heads).permute(1, 2, 0, 3)
        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)

        # Aggregate values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, self.features_dim)
        out = out.view(b, self.features_dim, h, w)  # Reshape to (batch_size, features_dim, height, width)
        # Final linear transformation
        # out = self.to_out(out)

        return out


# Example of using the modified CrossAttention module
batch_size = 3
features_dim = 256
height = 64
width = 64
embed_dim = 1536

# Input feature maps and single batch embedding
feature_maps = torch.randn(batch_size, features_dim, height, width)
embedding = torch.randn(1, embed_dim)  # Single embedding for all batches

# Create the CrossAttention layer
cross_attention = CrossAttention(features_dim, embed_dim)

# Forward pass
output = cross_attention(feature_maps, embedding)  # Use the single embedding across all batches
print(output.shape)  # Expected output: (3, 256, 64, 64)
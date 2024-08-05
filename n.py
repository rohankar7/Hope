import os
import torch

t = torch.load('./latents/latent_54.pt')
print(t.shape)
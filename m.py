# Install libs
# !pip install torchmultimodal-nightly

# import libs
import torch
import torchvision
import torchvision.transforms.functional as F

from torch import nn
from tqdm import tqdm
from torchmultimodal.diffusion_labs.modules.adapters.cfguidance import CFGuidance
from torchmultimodal.diffusion_labs.modules.losses.diffusion_hybrid_loss import DiffusionHybridLoss
from torchmultimodal.diffusion_labs.samplers.ddpm import DDPModule
from torchmultimodal.diffusion_labs.predictors.noise_predictor import NoisePredictor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import linear_beta_schedule, DiscreteGaussianSchedule
from torchmultimodal.diffusion_labs.transforms.diffusion_transform import RandomDiffusionSteps
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput

# Define diffusion schedule
schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))

# Define prediction target
from torchmultimodal.diffusion_labs.models.adm_unet.adm import adm_unet

unet = adm_unet(
    time_embed_dim = 32,
    embed_dim = 32,
    embed_name = 'context',
    predict_variance_value = True,
    image_channels = 1,
)

# Down scaling input blocks for unet
class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels, cond_channels):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels + cond_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x, c):
    _, _, w, h = x.size()
    c = c.expand(-1, -1, w, h)  # Shape conditional input to match image
    x = self.block(torch.cat([x,c], 1)) # Convolutions over image + condition
    x_small = self.pooling(x)   # Downsample output for next block
    return x, x_small

# Upscaling blocks on unet
class UpBlock(nn.Module):
  def __init__(self, inp, out):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(2 * inp, out, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out, out, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    self.upsample = nn.Upsample(scale_factor = 2)

  def forward(self, x, x_small):
    x_big = self.upsample(x_small)      # Upscale input back towards original size
    x = torch.cat((x_big, x), dim = 1)  # Join previous block with across block
    x = self.block(x)                   # Convolutions over image
    return x

class UNet(nn.Module):
  def __init__(self, time_size=32, digit_size=32, steps=1000):
    super().__init__()
    cond_size = time_size + digit_size
    self.conv = nn.Conv2d(1, 128, kernel_size=3, padding=1)
    self.down = nn.ModuleList([DownBlock(128, 256, cond_size), DownBlock(256, 512, cond_size)])
    self.bottleneck = DownBlock(512, 512, cond_size)
    self.up = nn.ModuleList([UpBlock(512, 256), UpBlock(256, 128)])
    # Define UNet

    self.time_projection = nn.Embedding(steps, time_size)
    self.prediction = nn.Conv2d(128, 1, kernel_size=3, padding=1)
    self.variance = nn.Conv2d(128, 1, kernel_size=3, padding=1)

  def forward(self, x, t, conditional_inputs):
    b, c, h, w = x.shape
    timestep = self.time_projection(t).view(b, -1, 1, 1)
    condition = conditional_inputs['context'].view(b, -1, 1, 1)
    condition = torch.cat([timestep, condition], dim=1)

    x = self.conv(x)
    outs = []
    for block in self.down:
      out, x = block(x, condition)
      outs.append(out)
    x, _ = self.bottleneck(x, condition)
    for block in self.up:
      x = block(outs.pop(), x)

    v = self.variance(x)
    p = self.prediction(x)

    return DiffusionOutput(p, v)

# Diffusion Module

unet = UNet(time_size=32, digit_size=32)
# Add support for classifier free guidance
unet = CFGuidance(unet, {'context':32}, guidance=2.0)
# Define evaluation
eval_steps = torch.linspace(0, 999, 250, dtype=torch.long)
model = DDPModule(unet, schedule, predictor, eval_steps)
# Define conditional embeddings
encoder = nn.Embedding(10, 32)

from torchvision.transforms import Compose, Resize, ToTensor, Lambda

# Transform data and use RandomDiffusionSteps to sample random noise
diffusion_transforms = RandomDiffusionSteps(schedule, batched=False)
transform = Compose([Resize(32),
                      ToTensor(),
                      Lambda(lambda x: 2*x - 1),
                      Lambda(lambda x: diffusion_transforms({'x':x}))])

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

train_dataset = FashionMNIST('fashion_mnist', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=2, pin_memory=True)

# Train model

epochs = 25

device = "cpu"
encoder.to(device)
model.to(device)

# Apply optimizer to diffusion model and encoder for joint training
optimizer = torch.optim.AdamW(
    [{"params": encoder.parameters()}, {"params": model.parameters()}], lr=0.0001
)
# Define loss
h_loss = DiffusionHybridLoss(schedule) # MSELoss()

encoder.train()
model.train()
for e in range(epochs):
  for sample in (pbar := tqdm(train_dataloader)):
    x, c = sample
    x0, xt, noise, t, c = x["x"].to(device), x["xt"].to(device), x["noise"].to(device), x["t"].to(device), c.to(device)
    optimizer.zero_grad()
    # Compute loss
    embedding = encoder(c)
    out = model(xt, t, {'context': embedding})
    loss = h_loss(out.prediction, noise, out.mean, out.log_variance, x0, xt, t)
    loss.backward()
    optimizer.step()

    pbar.set_description(f'{e+1}| Loss: {loss.item()}')

# Generate

def fashion_encoder(name, num=1):
    fashion_dict = {"t-shirt": 0, "pants": 1, "sweater": 2, "dress": 3, "coat": 4,
                    "sandal": 5, "shirt": 6, "sneaker": 7, "purse": 8, "boot": 9}
    idx = torch.as_tensor([fashion_dict[name] for _ in range(num)]).to(device)

    encoder.eval()
    with torch.no_grad():
        embed = encoder(idx)
    return embed

model.eval()

c = fashion_encoder("boot", 9)
noise = torch.randn(size=(9,1,32,32)).to(device)

with torch.no_grad():
    imgs = model(noise, conditional_inputs={"context": c})

img_grid = torchvision.utils.make_grid(imgs, 3)
img = F.to_pil_image((img_grid + 1) / 2)
img.resize((288, 288))
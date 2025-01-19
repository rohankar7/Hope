import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import datasets, transforms
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])
dataset = datasets.ImageFolder("images", transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))

def imsave(name,img):
	npimg = img.numpy()
	plt.imsave(name,np.transpose(npimg, (1, 2, 0)))

#imshow(torchvision.utils.make_grid(images))

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.eConv1 = nn.Conv2d(3,6,4)
		self.eConv2 = nn.Conv2d(6,12,5)
		self.ePool1 = nn.MaxPool2d(2,2)
		self.eConv3 = nn.Conv2d(12,24,5)
		self.ePool2 = nn.MaxPool2d(2,2)
		self.eF1 = nn.Linear(24*4*4,180)
		self.eMu = nn.Linear(180,180)
		self.eSigma = nn.Linear(180,180)

		self.dConvT1 = nn.ConvTranspose2d(180,200,4)
		self.dBatchNorm1 = nn.BatchNorm2d(200)
		self.dConvT2 = nn.ConvTranspose2d(200,120,6,2)
		self.dBatchNorm2 = nn.BatchNorm2d(120)
		self.dConvT3 = nn.ConvTranspose2d(120,60,6,2)
		self.dBatchNorm3 = nn.BatchNorm2d(60)
		self.dConvT4 = nn.ConvTranspose2d(60,3,5,1)

	def encode(self,x):
		x = self.eConv1(x)
		x = F.relu(x)
		x = self.eConv2(x)
		x = F.relu(x)
		x = self.ePool1(x)
		x = self.eConv3(x)
		x = F.relu(x)
		x = self.ePool2(x)
		x = x.view(x.size()[0], -1)
		x = self.eF1(x)
		mu = self.eMu(x)
		sigma = self.eSigma(x)
		return((mu,sigma))

	# From https://github.com/pytorch/examples/blob/master/vae/main.py
	def reparameterize(self,mu,sigma):
		std = torch.exp(0.5*sigma)
		eps = torch.randn_like(std)
		return (mu + eps*std)

	def decode(self,x):
		x = torch.reshape(x,(x.shape[0],180,1,1))
		x = self.dConvT1(x)
		x = self.dBatchNorm1(x)
		x = F.relu(x)
		x = self.dConvT2(x)
		x = self.dBatchNorm2(x)
		x = F.relu(x)
		x = self.dConvT3(x)
		x = self.dBatchNorm3(x)
		x = F.relu(x)
		x = self.dConvT4(x)
		x = torch.sigmoid(x)
		return(x)
		
	def forward(self,x):
		mu,sigma = self.encode(x)
		z = self.reparameterize(mu,sigma)
		x_gen = self.decode(z)
		return((x_gen,mu,sigma))
		
# From https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, x_gen, mu, sigma):
	#print(x.shape)
	#print(x_gen.shape)
	BCE = F.binary_cross_entropy(x_gen, x, reduction='sum')

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

	return BCE + KLD

vae = VAE()
vae.to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-4)

for epoch in range(200):

	running_loss = 0.0
	for i, data in enumerate(dataloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		images = data[0].to(device)
		#images = data[0]
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = vae(images)
		loss = loss_function(images, outputs[0], outputs[1], outputs[2])
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 500 == 499:    # print every 500 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

	PATH = 'vae_checkpoints/'
	torch.save(vae.state_dict(), PATH+str(epoch)+".pt")
	imsave("actual/" + str(epoch) + ".png", torchvision.utils.make_grid(images.cpu()))
	imsave("recon/" + str(epoch) + ".png",torchvision.utils.make_grid(outputs[0].detach().cpu()))

print('Finished Training')
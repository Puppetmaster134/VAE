import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


import matplotlib.pyplot as plt

class VAE(nn.Module):
	def __init__(self, hidden_size):
		super(VAE, self).__init__()


		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=3,
				kernel_size=(4,4),
				stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(2,stride=2),
			nn.Conv2d(
				in_channels=3,
				out_channels=3,
				kernel_size=(2,2),
				stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(2,stride=1)
		)

		self.encoder_fc1 = nn.Linear(300,20)
		self.encoder_fc2 = nn.Linear(300,20)

		self.decode_fc1 = nn.Linear(20,300)
		self.decode_fc2 = nn.Linear(300,784)

	def encode(self,input):
		# input = (batch_size * input_channels * input_height * input_width)
		h = self.conv(input).view(-1,300)
		return self.encoder_fc1(h), self.encoder_fc2(h)

	def reparameterize(self, mean, variance):
		std_dev = torch.exp(variance / 2)
		epsilon = torch.randn_like(std_dev)

		z = mean + epsilon * std_dev

		testcity = torch.zeros(20).to(std_dev.device)
		#idx = np.random.randint(0,20)
		idx = 0

		print('Boosted Feature:',idx)
		testcity[idx] = 12
		testcity = testcity * std_dev
		z = z + testcity
		return z

	def decode(self, z):
		h1 = F.relu(self.decode_fc1(z))
		return torch.sigmoid(self.decode_fc2(h1))

	def forward(self,input):
		#normal distributions for each feature
		mean,variance = self.encode(input)

		#Latent code z
		z = self.reparameterize(mean,variance)

		#output constructed from z
		x_prime = self.decode(z)#.view(-1,1,28,28)

		return x_prime, input, mean, variance

	# Reconstruction + KL divergence losses summed over all elements and batch
	def loss_function(self,recon_x, x, mu, logvar):
		BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


		return BCE, KLD

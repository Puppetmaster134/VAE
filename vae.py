import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


import matplotlib.pyplot as plt

class VAE(nn.Module):
	def __init__(self, hidden_size, image_height, image_width):
		super(VAE, self).__init__()

		#Number of input/output features
		self.output_features = image_height * image_width
		
		#Number of channels ConvNet will extract features with
		self.conv_channels = 3
		
		#The size of z (encoded variables)
		self.hidden_size = hidden_size
		
		#First Convolutional Layer Parameters
		self.conv1_size = 4
		self.conv1_stride = 1
		self.pool1_size = 2
		self.pool1_stride = 2
		
		#Second Convolutional Layer Parameters
		self.conv2_size = 2
		self.conv2_stride = 1
		self.pool2_size = 2
		self.pool2_stride = 1
		
		#Calculating the number of features extracted from our convolutional sequence
		conv1_output_size = self.calculate_conv_output(image_height,self.conv1_size, self.conv1_stride)
		pool1_output_size = self.calculate_pool_output(conv1_output_size, self.pool1_size, self.pool1_stride)
		conv2_output_size = self.calculate_conv_output(pool1_output_size,self.conv2_size, self.conv2_stride)
		pool2_output_size = self.calculate_pool_output(conv2_output_size, self.pool2_size, self.pool2_stride)
		seq_output_size = np.floor(pool2_output_size)
		self.conv_features = int(np.square(seq_output_size) * self.conv_channels)

		#Sequential Layer with the Convolutional Layers inside
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=self.conv_channels,
				kernel_size=(self.conv1_size, self.conv1_size),
				stride=self.conv1_stride),
			nn.ReLU(True),
			nn.MaxPool2d(self.pool1_size,stride=self.pool1_stride),
			nn.Conv2d(
				in_channels=self.conv_channels,
				out_channels=self.conv_channels,
				kernel_size=(self.conv2_size,self.conv2_size),
				stride=self.conv2_stride),
			nn.ReLU(True),
			nn.MaxPool2d(self.pool2_size,stride=self.pool2_stride)
		)
		
		self.encoder_fc1 = nn.Linear(self.conv_features,self.hidden_size)
		self.encoder_fc2 = nn.Linear(self.conv_features,self.hidden_size)

		self.decode_fc1 = nn.Linear(self.hidden_size,self.conv_features)
		self.decode_fc2 = nn.Linear(self.conv_features,self.output_features)

	def encode(self,input):
		h = self.conv(input).view(-1,self.conv_features)
		return self.encoder_fc1(h), self.encoder_fc2(h)

	def reparameterize(self, mean, variance):
		std_dev = torch.exp(variance / 2)
		epsilon = torch.randn_like(std_dev)
		z = mean + epsilon * std_dev
		return z

	def decode(self, z):
		h1 = F.relu(self.decode_fc1(z))
		return torch.sigmoid(self.decode_fc2(h1))

	def forward(self,input):
		#Means and Variances for each hidden feature
		mean,variance = self.encode(input)

		#Latent State as sampled values (aka. z)
		z = self.reparameterize(mean,variance)

		#The output we decoded from z
		x_prime = self.decode(z)#.view(-1,1,28,28)

		return x_prime, input, mean, variance

	
	def loss_function(self,recon_x, x, mu, logvar):		
		#Binary Cross Entropy loss function
		BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.output_features), reduction='sum')
		
		#Kullback-Leibler (KL) Divergence loss function
		#Calculates the loss between our encoded distribution and the Normal distribution
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

		return BCE, KLD
	
	#Given some parameters, calculate the number of features returned by a ConvLayer
	#http://cs231n.github.io/convolutional-networks/
	def calculate_conv_output(self, input_width, kernel_width, stride = 1, padding_size = 0):
		return ((input_width + (2 * padding_size) - kernel_width) / stride) + 1
	
	#Given some parameters, calculate the number of features returned by a Pooling Layer
	#http://cs231n.github.io/convolutional-networks/
	def calculate_pool_output(self, input_width, kernel_width, stride = 1):
		return ((input_width - kernel_width) / stride) + 1

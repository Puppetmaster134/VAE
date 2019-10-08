import torch
from torch import nn

class VAE(nn.Module):
	def __init__(self, path_to_save, model_name):
		super(VAE, self).__init__()
		self.path_to_save = path_to_save
		self.model_name = model_name
		
	# Train our model and save it
	def train(self, dataset):
		torch.save(self.state_dict(), self.path_to_save + self.model_name)
	
	# Attempt an inference with our existing model
	def infer(self):
		pass
		
	# Attempt to load an existing model from disk
	def attempt_load(self):
		self.load_state_dict(torch.load(self.path_to_save + self.model_name))
		
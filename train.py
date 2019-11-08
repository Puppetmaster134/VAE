from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from vae import VAE


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--model',default='VAE',help='Name of model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
args = parser.parse_args()

# Set up CUDA if it's not disabled via argument
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

MODEL_STORE_PATH = 'C:\\Users\\Brian\\Documents\\Projects\\vae\\Models\\'
MODEL_NAME = f"{args.model}.ckpt"

torch.manual_seed(args.seed)

# Set up our training and testing sets (MNIST)
# 60000 elements

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    
num_items = train_dataset.data.size(0)
image_height = train_dataset.data.size(1)
image_width = train_dataset.data.size(2)
latent_features = 20


# Initialize our model
model = VAE(latent_features, image_height, image_width).to(device)


# Setup Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(0,args.epochs):
    total_loss = 0
    if __name__ == '__main__':
        for batch_idx, (data, _) in enumerate(train_loader):
            
            #Moves our data to our primary device (CPU or GPU)
            data = data.to(device)
            
            #Reset the gradients
            optimizer.zero_grad()
            
            #Our model returns the reconstruction, the input, and our encoder hidden state as means and variances
            x_prime, x, mean, variance = model(data)
            
            #Get the Binary Cross Entropy and KL Divergence
            BCE, KLD = model.loss_function(x_prime,x,mean,variance)
            
            #Add up Binary Cross Entropy and KL Divergence to get our loss
            loss = BCE + KLD
            
            #Backpropagation!
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
            #Some print stuff
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

#Save the model
#torch.save(model.state_dict(), f'{MODEL_STORE_PATH}/{MODEL_NAME}')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
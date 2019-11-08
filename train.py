from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from vae import VAE
import dataset_utility as dutil



parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--model',default='VAE',help='Name of model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default=None,help='Path to dataset to load')
args = parser.parse_args()

# Set up CUDA if it's not disabled via argument
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

MODEL_STORE_PATH = 'C:\\Users\\Brian\\Documents\\Projects\\vae\\Models\\'
MODEL_NAME = f"{args.model}.ckpt"

torch.manual_seed(args.seed)

#Load our dataset
train_dataset,img_width = dutil.byPath(args.dataset) if args.dataset else dutil.MNIST()

#Set up our data and parameters
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
num_items = len(train_dataset)
image_height = img_width
image_width = img_width
latent_features = 20


#Initialize the VAE
model = VAE(latent_features, image_height, image_width).to(device)


# Setup Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

kld_history = []
bce_history = []

if __name__ == '__main__':
    for epoch in range(0,args.epochs):
        total_loss = 0
        total_kld = 0
        total_bce = 0
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

            total_loss += loss.item()
            total_kld += KLD.item()
            total_bce += BCE.item()

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

        kld_history.append(total_kld / len(train_loader.dataset))
        bce_history.append(total_bce / len(train_loader.dataset))

    #Save the model
    torch.save(model.state_dict(), f'{MODEL_STORE_PATH}/{MODEL_NAME}')

    plt.plot(range(0,args.epochs), kld_history, 'r--')
    plt.plot(range(0,args.epochs), bce_history, 'b-')
    plt.legend(['KL Divergence', 'Binary Cross-Entropy'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.xticks(range(0,args.epochs))

    plt.show()

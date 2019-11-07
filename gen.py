from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from vae import VAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true', default=False, help='determines whether we want to train or infer')
args = parser.parse_args()

# Set up CUDA if it's not disabled via argument
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}






MODEL_STORE_PATH = 'C:\\Users\\Brian\\Documents\\Projects\\vae\\Models\\'
MODEL_NAME = "VAE.ckpt"

model = VAE(20).to('cuda')
model.load_state_dict(torch.load(f'{MODEL_STORE_PATH}/{MODEL_NAME}'))
model.eval()

print(model)

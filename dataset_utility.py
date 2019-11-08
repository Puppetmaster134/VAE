from torchvision import datasets, transforms
import torch

def MNIST(train=True):
    return datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor()), 28

def byPath(path):
    return datasets.ImageFolder(path, transform=transforms.Compose([
            transforms.Resize(48),
            transforms.RandomCrop(48),
            transforms.ToTensor()])), 48

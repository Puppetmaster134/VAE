from torchvision import datasets, transforms

def MNIST(train=True):
    return datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor())
    
def ByPath(path):
    pass
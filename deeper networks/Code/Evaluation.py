import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import project_functions as pf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 25

transform_cifar = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

transform_mnist = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.Grayscale(3),
    transforms.ToTensor()
])


# load datasets 
test_mnist = torchvision.datasets.MNIST('./MNIST/', train=False, transform=transform_mnist)
test_cifar = torchvision.datasets.CIFAR10('./CIFAR/', train=False, transform=transform_cifar)

mnist_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)
cifar_loader = DataLoader(dataset=test_cifar, batch_size=batch_size, shuffle=False)

# load best models
googlenet_mnist = torch.load('./models/GoogLeNet_train_MNIST.pth').to(device)
googlenet_cifar = torch.load('./models/GoogLeNet_train_CIFAR.pth').to(device)

resnet_mnist = torch.load('./models/ResNet34_train_MNIST.pth').to(device)
resnet_cifar = torch.load('./models/ResNet34_train_CIFAR.pth').to(device)

vgg_mnist = torch.load('./models/VGG16_train_MNIST.pth').to(device)
vgg_cifar = torch.load('./models/VGG16_train_CIFAR.pth').to(device)


# compute accuracy of best models on datasets
pf.test_model(googlenet_cifar, 'GoogLeNet_CIFAR', test_loader=cifar_loader, batch_size=batch_size)
pf.test_model(googlenet_mnist, 'GoogLeNet_MNIST', test_loader=mnist_loader, batch_size=batch_size)
pf.test_model(resnet_cifar, 'ResNet_CIFAR', test_loader=cifar_loader, batch_size=batch_size)
pf.test_model(resnet_mnist, 'ResNet_MNIST', test_loader=mnist_loader, batch_size=batch_size)
pf.test_model(vgg_cifar, 'VGG_CIFAR', test_loader=cifar_loader, batch_size=batch_size)
pf.test_model(vgg_mnist, 'VGG_MNIST', test_loader=mnist_loader, batch_size=batch_size)


# display single case image and classification of image by selected model
pf.single_image(vgg_cifar, test_cifar, index = 1002, 'VGG_CIFAR_image')
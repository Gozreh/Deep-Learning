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


model_save_dir = './models/'
stats_save_dir='./stats/'

# create folder if not exist
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
if not os.path.exists(stats_save_dir):
    os.mkdir(stats_save_dir)


#---------------------------------
# Parameters
#---------------------------------

batch_size=25
epochs=15
learning_rate=0.0001

#---------------------------------
# Data Loading
#---------------------------------

transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

# load train and test datasets 
data = torchvision.datasets.CIFAR10('./CIFAR/', transform=transform)
train_data, val_data = random_split(data, [40000,10000])

test_data = torchvision.datasets.CIFAR10('./CIFAR/', train=False, transform=transform)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


#---------------------------------
# Define Model and other Criteria
#---------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import untrained GoogLeNet model
model = torchvision.models.resnet34(pretrained=False).to(device)

#Cross Entropy Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Initialise the Optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)


#---------------------------------
# Train and Validate Model
#---------------------------------

# train model
train_hist = pf.train_model(model, train_loader, val_loader, criterion, optimiser, epochs, batch_size, 'ResNet34_train_CIFAR')

# show training statistics
pf.loss_train_hist(train_hist, save=True, path='./stats/' + 'ResNet34_loss_CIFAR' + '.png')
pf.acc_train_hist(train_hist, save=True, path='./stats/' + 'ResNet34_acc_CIFAR' + '.png')


#---------------------------------
# Evaluate Model
#---------------------------------

#model = torch.load('./models/GoogLeNet_train_MNIST.pth').to(device)


#pf.test_model(model, test_loader=test_loader)
#pf.single_image(model, test_data, 11, 'GoogLeNet_MNIST_image')


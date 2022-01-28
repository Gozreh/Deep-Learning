import time
import pickle
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

#---------------------------------
# Useful Functions
#---------------------------------

def loss_train_hist(hist, show=False, save=False, path='Train_hist_loss.png'):

    """Loss tracker.
    
    Arguments:
        hist {[dict]} -- Tracking variables

    Keyword Arguments:
        show {bool} -- If to display the figure (default: {False})
        save {bool} -- If to store the figure (default: {False})
        path {str} -- path to store the figure (default: {'Train_hist.png'}) """
    
    x = range(1,len(hist['train_losses'])+1)

    y1 = hist['train_losses']
    y2 = hist['val_losses']

    plt.plot(x, y1, label='training loss')
    plt.plot(x, y2, label='validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def acc_train_hist(hist, show=False, save=False, path='Train_hist_acc.png'):

    """Accuracy tracker.
    
    Arguments:
        hist {[dict]} -- Tracking variables

    Keyword Arguments:
        show {bool} -- If to display the figure (default: {False})
        save {bool} -- If to store the figure (default: {False})
        path {str} -- path to store the figure (default: {'Train_hist.png'}) """
    
    x = range(1,len(hist['train_losses'])+1)

    y1 = hist['train_accuracy']
    y2 = hist['val_accuracy']

    plt.plot(x, y1, label='train accuracy')
    plt.plot(x, y2, label='validation accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend(loc=0)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def train_model(model, train_loader, val_loader, criterion, optimiser, epochs, batch_size, filename, save=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tracking variables
    train_hist = {}
    train_hist['train_losses'] = []
    train_hist['val_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    train_hist['train_accuracy'] = []
    train_hist['val_accuracy'] = []

    start_time = time.time()

    for epoch in range(epochs):

        # set model to train mode and declare tracking variables
        model.train()
        Train_loss = []
        Val_loss = []
        Acc_train_model = []
        Acc_val_model =[]
        epoch_start_time = time.time()

        for image, label in tqdm(train_loader):
            image=image.to(device)
            label=label.to(device)
        
            # train model and compute loss and accuracy
            output_value = model(image)
            loss = criterion(output_value, label)

            _, preds = torch.max(output_value, 1)
            acc = (preds == label).float().sum().item()
            acc = acc/batch_size

               
            # back propagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # store loss and accuracy
            Train_loss.append(loss.item())
            Acc_train_model.append(acc)

        # set to evaluation mode for validation and declare acc_best
        model.eval()
        acc_best = 0

        for image, label in tqdm(val_loader):
            image=image.to(device)
            label=label.to(device)

            # train model and compute loss and accuracy
            output_value = model(image)
            loss = criterion(output_value, label)

            _, preds = torch.max(output_value, 1)
            acc = (preds == label).float().sum().item()
            acc = acc/batch_size

            # store accuracy
            Val_loss.append(loss.item())
            Acc_val_model.append(acc)

            # store model with best validation accuracy
            if acc > acc_best:
                acc_best = acc
                model_best = model


        epoch_loss = np.mean(Train_loss + Val_loss)
        train_loss = np.mean(Train_loss)
        val_loss = np.mean(Val_loss)
        epoch_train_accuracy = np.mean(Acc_train_model)
        epoch_val_accuracy = np.mean(Acc_val_model)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print("Epoch %d of %d with %.2f s" % (epoch + 1, epochs, per_epoch_ptime))
        print("Epoch loss: %.8f" % (epoch_loss))
        print("Epoch train accuracy: %.8f" % (epoch_train_accuracy))
        print("Epoch validation accuracy: %.8f" % (epoch_val_accuracy))

        # record the loss and accuracy for every epoch
        train_hist['train_losses'].append(train_loss)
        train_hist['val_losses'].append(val_loss)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        train_hist['train_accuracy'].append(epoch_train_accuracy)
        train_hist['val_accuracy'].append(epoch_val_accuracy)


    # save model
    if save:
        torch.save(model_best, './models/' + filename + '.pth')

    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)


    # end of training
    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
        np.mean(train_hist['per_epoch_ptimes']), epochs, total_ptime))

    print("Training finish!... save training results")

    with open('./stats/' + filename + '.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    return train_hist

def train_model_aux(model, train_loader, val_loader, criterion, optimiser, epochs, batch_size, filename, save=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tracking variables
    train_hist = {}
    train_hist['train_losses'] = []
    train_hist['val_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    train_hist['train_accuracy'] = []
    train_hist['val_accuracy'] = []

    start_time = time.time()

    for epoch in range(epochs):

        # set model to train mode and declare tracking variables
        model.train()
        Train_loss = []
        Val_loss = []
        Acc_train_model = []
        Acc_val_model =[]
        epoch_start_time = time.time()

        for image, label in tqdm(train_loader):
            image=image.to(device)
            label=label.to(device)
        
            # train model and compute loss and accuracy. 
            output_value, aux_output, _= model(image)
            loss1 = criterion(output_value, label)
            loss2 = criterion(aux_output, label)
            loss = loss1 + 0.3*loss2

            _, preds = torch.max(output_value, 1)
            acc = (preds == label).float().sum().item()
            acc = acc/batch_size

               
            # back propagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # store loss and accuracy
            Train_loss.append(loss.item())
            Acc_train_model.append(acc)

        # set to evaluation mode for validation and declare acc_best
        model.eval()
        acc_best = 0

        for image, label in tqdm(val_loader):
            image=image.to(device)
            label=label.to(device)

            # train model and compute loss and accuracy
            output_value = model(image)
            loss = criterion(output_value, label)

            _, preds = torch.max(output_value, 1)
            acc = (preds == label).float().sum().item()
            acc = acc/batch_size

            # store accuracy
            Val_loss.append(loss.item())
            Acc_val_model.append(acc)

            # store model with best validation accuracy
            if acc > acc_best:
                acc_best = acc
                model_best = model


        epoch_loss = np.mean(Train_loss + Val_loss)
        train_loss = np.mean(Train_loss)
        val_loss = np.mean(Val_loss)
        epoch_train_accuracy = np.mean(Acc_train_model)
        epoch_val_accuracy = np.mean(Acc_val_model)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print("Epoch %d of %d with %.2f s" % (epoch + 1, epochs, per_epoch_ptime))
        print("Epoch loss: %.8f" % (epoch_loss))
        print("Epoch train accuracy: %.8f" % (epoch_train_accuracy))
        print("Epoch validation accuracy: %.8f" % (epoch_val_accuracy))

        # record the loss and accuracy for every epoch
        train_hist['train_losses'].append(train_loss)
        train_hist['val_losses'].append(val_loss)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        train_hist['train_accuracy'].append(epoch_train_accuracy)
        train_hist['val_accuracy'].append(epoch_val_accuracy)


    # save model
    if save:
        torch.save(model_best, './models/' + filename + '.pth')

    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)


    # end of training
    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
        np.mean(train_hist['per_epoch_ptimes']), epochs, total_ptime))

    print("Training finish!... save training results")

    with open('./stats/' + filename + '.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    return train_hist

def test_model(model, model_name, test_loader, batch_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    Acc_test_model = []

    for image, label in tqdm(test_loader):

        image=image.to(device)
        label=label.to(device)

        # get model predictions
        output_value = model(image)

        _, preds = torch.max(output_value, 1)
        acc = (preds == label).float().sum().item()
        acc = acc/batch_size

        # store accuracy
        Acc_test_model.append(acc)
    
    test_accuracy = np.mean(Acc_test_model)

    print(f'{model_name} test accuracy: ', test_accuracy)

def single_image(model, test_data, index, filename, save=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # prepare data loader with single image
    item = test_data[index]
    single_loader = DataLoader(dataset=item, batch_size=1, shuffle=False)
    image, label = single_loader
    image = image.to(device)
    label = label.item()

    # get prediction
    output_value = model(image)

    _, preds = torch.max(output_value, 1)
    preds = preds.item()

    ground_truth = item[0]

    resize = transforms.Compose([transforms.Resize((28, 28))])
    ground_truth = resize(ground_truth).permute(1,2,0)
    
    if save:
        plt.imshow(ground_truth)
        plt.title(f'Predicted Class: {preds} -- Actual Class: {label}')
        plt.savefig('./stats/' + filename + f'{index}.png')


#---------------------------------
# Model Architectures
#---------------------------------

#------------
# GoogLeNet
#------------

"""
The model consists of:
* Conv2d + BatchNorm2d + ReLU block class
* Inception block class
* Auxiliary classifier class
* Model class

"""
class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=False, num_classes=1000):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x

class Inception_block(nn.Module):

    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))



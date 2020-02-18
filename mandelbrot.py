from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from activations import MandelbrotActivation
from data_processor import train_set, test_set

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

class NeuralModel(nn.Module):
    def __init__(self, custom = True):
        super().__init__()
        self.custom = custom
        num_channels = 8

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=5).to(device)
        self.mpool1= nn.MaxPool2d(2).to(device)
        self.b1 = nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True).to(device)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1).to(device)
        self.b2 = nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True).to(device)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=5).to(device)
        self.mpool3 = nn.MaxPool2d(2)
        self.b3 = nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True).to(device)

        self.fc1 = nn.Linear(num_channels * 4 ** 2, 20).to(device)
        self.fc2 = nn.Linear(20, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.b1(x)

        if self.custom:
            activation = MandelbrotActivation(x.shape).to(device)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)
        
        x = self.conv2(x)
        x = self.b1(x)

        if self.custom:
            activation = MandelbrotActivation(x.shape).to(device)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        x = self.conv3(x)
        x = self.mpool3(x)
        x = self.b3(x)

        if self.custom:
            activation = MandelbrotActivation(x.shape).to(device)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        x = self.fc1(x.view(x.shape[0], -1))

        if self.custom:
            activation = MandelbrotActivation(x.shape).to(device)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        x = self.fc2(x)

        if self.custom:
            activation = MandelbrotActivation(x.shape).to(device)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        output = x
        
        return output

class FCModel(nn.Module):

    def __init__(self, custom = True):
        super(FCModel, self).__init__()

        self.fc1 = nn.Linear(in_features = 784, out_features = 100)
        self.fc2 = nn.Linear(in_features = 100, out_features = 100)
        self.fc3 = nn.Linear(in_features = 100, out_features = 50)
        self.fc4 = nn.Linear(in_features = 50, out_features = 10)
        self.custom = custom


    def forward(self, x):
        x = self.fc1(x)
        b = nn.BatchNorm1d(100)
        x = b(x)

        if self.custom:
            activation = MandelbrotActivation(100)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)
        
        x = self.fc2(x)
        b = nn.BatchNorm1d(100)
        x = b(x)

        if self.custom:
            activation = MandelbrotActivation(100)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        x = self.fc3(x)
        b = nn.BatchNorm1d(50)
        x = b(x)

        if self.custom:
            activation = MandelbrotActivation(50)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        x = self.fc4(x)
        b = nn.BatchNorm1d(10)
        x = b(x)

        if self.custom:
            activation = MandelbrotActivation(10)
            x = activation(x)
        else:
            activation = nn.ReLU()
            x = activation(x)

        output = x

        return output 

def train_model(model, train_data, epochs = 10):
    loss_trace = []
    criterion = nn.CrossEntropyLoss()
    learning_rate = .01 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = epochs
    model.train()
    
    model.to(device)
    
    print("started training ...")

    for epoch in range(n_epochs):
    	if epoch / 10 ==0:
    		learning_rate /= 2
    		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    	epoch_loss = 0.0 
    	for batch in train_data:
    		batch_images, batch_labels = batch

    		batch_images = batch_images.to(device)
    		batch_labels = batch_labels.to(device)

    		batch_output = model(batch_images)
            
    		loss = criterion(batch_output, batch_labels)
            
    		optimizer.zero_grad()
    		loss.backward()
    		epoch_loss += loss.item()
    		optimizer.step()
        
    	print("the loss after processing this epoch is: ", epoch_loss)
    	loss_trace.append(epoch_loss)
    print("Training completed.")
    print("=*="*20)
    return model, loss_trace


if __name__ == "__main__": 
	batch_size = 512
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
	
	neural_model = NeuralModel(custom = False)
	trained_model, loss_trace = train_model(neural_model, train_loader)
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

        num_channels = 8
        if custom:
        	self.activation_function = MandelbrotActivation()
        else:
        	self.activation_function = nn.ReLU(inplace = True)
        
        self.conv = nn.Sequential(

            nn.Conv2d(1, num_channels, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            self.activation_function,

            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            self.activation_function,

            nn.Conv2d(num_channels, num_channels, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            self.activation_function

        )
        self.fc1 = nn.Linear(num_channels * 4 ** 2, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        convolved = self.conv(x)
        after_fc1 = self.activation_function(self.fc1(convolved.view(convolved.size(0), -1)))
        output = self.fc2(after_fc1)
        return output

def train_model(model, train_data, epochs = 10):
    loss_trace = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = epochs
    model.train()
    
    model.to(device)
    
    print("started training ...")

    for epoch in range(n_epochs):
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
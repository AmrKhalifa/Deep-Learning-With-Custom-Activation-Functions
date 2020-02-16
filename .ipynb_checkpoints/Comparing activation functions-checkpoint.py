#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
from mandelbrot import NeuralModel, train_model
from data_processor import train_set, test_set
from activations import MandelbrotActivation
import matplotlib.pyplot as plt 


# In[ ]:


use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

batch_size = 512
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


# In[ ]:


custom_neural_model = NeuralModel(custom = True)
custom_trained_model, loss_trace = train_model(custom_neural_model, train_loader, epochs = 10)

regular_neural_model = NeuralModel(custom = False)
regular_trained_model, regular_loss_trace = train_model(regular_neural_model, train_loader, epochs = 10)


# In[ ]:


plt.plot(loss_trace)
plt.plot(regular_loss_trace)
plt.show() 


# In[ ]:





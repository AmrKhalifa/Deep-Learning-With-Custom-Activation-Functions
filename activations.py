import torch
import torch.nn as nn

class MandelbrotActivation(nn.Module): 

	def __init__(self, lmbda = None):
		super(MandelbrotActivation, self).__init__()
		self.in_features = in_features

		if lmbda == None:
			self.lmbda = torch.nn.Parameter(torch.tensor(0.0))
		else:
			self.lmbda = torch.nn.Parameter(torch.tensor(lmbda))

		self.lmbda.requires_grad = True

	def forward(self, x):

		return self.lmbda*(x)*(1-x)

class FCMandelbrotActivation(nn.Module): 

	def __init__(self, lmbda = None):
		super(FCMandelbrotActivation, self).__init__()
		
		self.in_features = in_features
		if lmbda == None:
			self.lmbda = torch.nn.Parameter(torch.zeros(in_features, 1))
		else:
			self.lmbda = torch.nn.Parameter(torch.randn(in_features, 1))

		self.lmbda.requires_grad = True

	def forward(self, x):

		return self.lmbda*(x)*(1-x)
import torch
import torch.nn as nn

class MandelbrotActivation(nn.Module): 

	def __init__(self, lmbda = None):
		super(MandelbrotActivation, self).__init__()

		if lmbda == None:
			self.lmbda = torch.nn.Parameter(torch.tensor(0.0))
		else:
			self.lmbda = torch.nn.Parameter(torch.tensor(lmbda))

		self.lmbda.requires_grad = True

	def forward(self, x):

		return self.lmbda*(x)*(1-x)
import torch
import torch.nn as nn

class MandelbrotActivation(nn.Module): 

	def __init__(self, num_parameters ,lmbda = None):
		super(MandelbrotActivation, self).__init__()

		self.num_parameters = num_parameters
		
		if lmbda == None:
			self.lmbda = torch.nn.Parameter(torch.Tensor(self.num_parameters).fill_(0.0))
		else:
			self.lmbda = torch.nn.Parameter(torch.Tensor(self.num_parameters).fill_(lmbda))

		self.lmbda.requires_grad = True

	def forward(self, x):

		return self.lmbda*(x)*(1-x)
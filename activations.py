import torch
import torch.nn as nn

class MandelbrotActivation(nn.Module): 

	def __init__(self, shapes):
		super(MandelbrotActivation, self).__init__()

		self.lmbda = torch.nn.Parameter(torch.randn(shapes)*.25 + .5)
		self.lmbda.requires_grad = True

	def forward(self, x):

		return self.lmbda*(x)*(1-x)
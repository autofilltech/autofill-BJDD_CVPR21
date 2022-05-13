import torch
import torch.nn as nn

class ReduceAdd(nn.Module):
	def __init__(self, *args, **kwargs):
		super(ReduceAdd, self).__init__()
		self.args = args
		
	def forward(self, x):
		y = None
		for a in self.args:
			if not isinstance(a, torch.Tensor): a = a(x)
			y = a if y is None else y + a
		return y

class ReduceMul(nn.Module):
	def __init__(self, *args, **kwargs):
		super(ReduceMul, self).__init__()
		self.args = args
		
	def forward(self, x):
		y = None
		for a in self.args:
			if not isinstance(a, torch.Tensor): a = a(x)
			y = a if y is None else y * a
		return y

class ReduceCat(nn.Module):
	def __init__(self, *args, **kwargs):
		super(ReduceCat, self).__init__()
		self.args = args
		self.dim = kwargs["dim"] if "dim" in kwargs else 0
		
	def forward(self, x):
		outputs = []
		for a in self.args:
			if not isinstance(a, torch.Tensor): a = a(x)
			outputs.append(a)
		return torch.cat(outputs, dim=self.dim)


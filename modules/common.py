import torch
import torch.nn as nn
import numpy as np

class SeparableConv2d(nn.Sequential):
	def __int__(self, features_in, features_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
		params = [
			nn.Conv2d(
				features_in, features_in, kernel_size, 
				stride, padding, dilation, 
				groups=channels_in, bias=bias),
			nn.Conv2d(
				features_in, features_out, 1,
				stride=1, padding=0, dilation=1,
				bias=bias)]
		if bn:
			params.append(bn())
		if act == nn.PReLU:
		
		super(SeparableConv2D, self).__init__(

class View(nn.Module):
	def __init__(self, *args):
		super(View, self).__init__()
		self.shape = args
	
	def forward(self, x):
		return x.view(*self.shape)

class Identity(nn.Module):
	def forward(self, x):
		return x

class MuxAdd(nn.Module):
	def __init__(self, *args, **kwargs):
		super(MuxSum, self).__init__()
		self.mux = args
		
	def forward(self, x):
		outputs = []
		for module in self.mux:
			outputs.append(module(x))
		return outputs.sum()

class MuxMul(nn.Module):
	def __init__(self, *args, **kwargs):
		super(MuxSum, self).__init__()
		self.mux = args
		
	def forward(self, x):
		outputs = []
		for module in self.mux:
			outputs.append(module(x))
		return np.prod(outputs)

class MuxSum(nn.Module):
	def __init__(self, *args, **kwargs):
		super(MuxSum, self).__init__()
		self.mux = args
		self.dim = kwargs["dim"] if "dim" in kwargs else 0
		
	def forward(self, x):
		outputs = []
		for module in self.mux:
			outputs.append(module(x))
		return torch.sum(outputs, dim=self.dim)

class MuxCat(nn.Module):
	def __init__(self, *args, **kwargs):
		super(MuxCat, self).__init__()
		self.mux = args
		self.dim = kwargs["dim"] if "dim" in kwargs else 0
		
	def forward(self, x):
		outputs = []
		for module in self.mux:
			outputs.append(module(x))
		return torch.cat(outputs, dim=self.dim)


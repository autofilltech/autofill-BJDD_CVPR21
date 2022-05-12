import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Squeeze and Excitation Layer
class SELayer(nn.Module):
	def __init__(self, channels, reduction = 16, bias = False):
		super(SELayer, self).__init__()
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.body = nn.Sequential(
				nn.Linear(channels, channels // reduction, bias = bias),
				nn.ReLU(inplace = True),
				nn.Linear(channels // reduction, channels, bias = bias),
				nn.Sigmoid())
	def forward(self, x):
		b,c,_,_ = x.size()
		y = self.pool(x).view(b, c)
		y = self.body(y).view(b, c, 1, 1)
		return x + y.expand_as(x)

# Spatial Attention Layer
class SALayer(nn.Module):
	def __init__(self, kernel_size = 3, bias = False):
		super(SALayer, self).__init__()
		self.body = nn.Sequential(
				FeaturePool2d(),
				nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = bias)
				nn.Sigmoid())

# Attention Guided Residual Block
class AttResBlock(nn.Module):
	def __init__(self, channels = 32, expand = None, dilation = 1, bias = True):
		super(AttResBlock, self).__init__()
		if expand is None: expand = channels * 2
		self.datt = SELayer(channels)
		self.body = nn.Sequential(
			nn.Conv2d(channels, expand, 1),
			nn.BatchNorm2d(expand),
			nn.ReLU(inplace = True),
			nn.Conv2d(expand, expand, kernel_size = 3, dilation = dilation, padding = dilation),
			nn.BatchNorm2d(expand),
			nn.ReLU(inplace = True),
			nn.Conv2d(expand, channels, 1))
		
	def forward(self, x):
		a = self.datt(x)
		y = self.body(x)
		return x + y + a

class FeaturePool2d(nn.Module):
	def forward(self, x):
		xmax,_ = torch.max(x, dim=-3, keep_dim=True)
		xmean = torch.mean(x, dim=-3, keep_dim=True)
		return torch.cat((xmax, xmean), dim=1)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Squeeze and Excitation Layer
class SqueezeExcitationLayer(nn.Module):
	def __init__(self, channels, reduction = 16, bias = False):
		super(SqueezeExcitationLayer, self).__init__()
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
class SpatialAttentionBlock(nn.Module):
	def __init__(self, kernel_size = 3, bias = False):
		super(SpatialAttentionBlock, self).__init__()
		self.body = nn.Sequential(
				MuxCat(
					ChannelMaxPool2d(),
					ChannelAvgPool2d(),
					dim = -3),
				nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = bias)
				nn.Sigmoid())

# Attention Guided Residual Block
class AttentionResBlock(nn.Sequential):
	def __init__(self, channels = 32, expand = 64):
		super(AttentionResBlock, self).__init__(
			MuxAdd(
				Identity(),
				SqueezeExcitationLayer(channels),
				nn.Sequential(
					nn.Conv2d(channels, expand, 1),
					nn.BatchNorm2d(expand),
					nn.ReLU(inplace = True),
					nn.Conv2d(expand, expand, kernel_size = 3),
					nn.BatchNorm2d(expand),
					nn.ReLU(inplace = True),
					nn.Conv2d(expand, channels, 1))))
		

class ChannelMaxPool2d(nn.Module):
	def forward(self, x):
		xmax,_ = torch.max(x, dim=-3, keep_dim=True)
		return xmax

class ChannelAvgPool2d(nn.Module):
	def forward(self, x):
		xmean = torch.mean(x, dim=-3, keep_dim=True)
		return xmean

class ChannelAttentionBlock(nn.Sequential):
	def __init__(self, channels, reduction=8, bias=False):
		super(ChannelAttentionBlock, self).__init__(
			MuxAdd(
				nn.Sequential(
					nn.MaxPool2d(2),
					nn.Conv2d(channels, channels//reduction, 1, bias=bias),
					nn.ReLU(inplace=True),
					nn.Conv2d(channels//reduction, channels, 1, bias=bias)),
				nn.Sequential(
					nn.AvgPool2d(2),
					nn.Conv2d(channels, channels//reduction, 1, bias=bias),
					nn.Relu(inplace=True),
					nn.Conv2d(channels//reduction, channels, 1, bias=bias))),
			nn.Sigmoid())


# Oc = C(x) * x
# y = S(Oc) * Oc + x
class RCSABlock(nn.Sequential):
	def __init__(self, channels):
		super(RCSABlock, self),__init__(
			MuxAdd(
				Identity(),
				nn.Sequential(
					MuxMul(
						ChannelAttentionBlock(channels),
						Identity),
					MuxMul(
						SpatialAttentionBlock(),
						Identity)))


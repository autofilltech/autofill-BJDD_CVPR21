import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.common import *
from modules.reduce import *

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
		return x * y.expand_as(x)

# Spatial Attention Layer
class SpatialAttentionLayer(nn.Sequential):
	def __init__(self, kernel_size = 3, bias = False):
		super(SpatialAttentionLayer, self).__init__(
			ReduceCat(
				ChannelMaxPool2d(),
				ChannelAvgPool2d(),
				dim = -3),
			nn.Conv2d(2, 1, kernel_size, padding = (kernel_size-1) // 2, bias = bias),
			nn.Sigmoid())

class SpatialAttentionBlock(nn.Sequential):
	def __init__(self, channels, kernel_size = 3, bias = False):
		super(SpatialAttentionBlock, self).__init__(
			ReduceMul(
				SpatialAttentionLayer(kernel_size, bias),
				nn.Conv2d(channels, channels, 3, padding=1)))

# Attention Guided Residual Block
class AttentionResBlock(nn.Sequential):
	def __init__(self, channels = 32, expand = 2):
		super(AttentionResBlock, self).__init__(
			ReduceAdd(
				Identity(),
				ReduceAdd(
					SqueezeExcitationLayer(channels),
					nn.Sequential(
						nn.Conv2d(channels, channels * expand, 1),
						nn.BatchNorm2d(channels*expand),
						nn.LeakyReLU(inplace = True),
						SeparableConv2d(channels*expand, channels*expand, 
							kernel_size=3, padding=1),
						nn.BatchNorm2d(channels*expand),
						nn.LeakyReLU(inplace = True),
						nn.Conv2d(channels*expand, channels, 1)))))

class ChannelMaxPool2d(nn.Module):
	def forward(self, x):
		xmax,_ = torch.max(x, dim=-3, keepdim=True)
		return xmax

class ChannelAvgPool2d(nn.Module):
	def forward(self, x):
		xmean = torch.mean(x, dim=-3, keepdim=True)
		return xmean

class ChannelAttentionLayer(nn.Sequential):
	def __init__(self, channels, reduction=8, bias=False):
		super(ChannelAttentionLayer, self).__init__(
			ReduceAdd(
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
			ResBlock(channels, channels, 3),
			ResBlock(channels, channels, 3),
			ReduceAdd(
				Identity(),
				nn.Sequential(
					ReduceMul(
						ChannelAttentionLayer(channels),
						Identity()),
					ReduceMul(
						SpatialAttentionLayer(),
						Identity()))))

# TODO make number of inputs variable.. ModuleList, register_parameter
class CrossAttentionBlock(nn.Module):
	def __init__(self, channels):
		super(CrossAttentionBlock, self).__init__()
		self.scale = channels ** -0.5
		self.normA = LayerNorm2d(channels)
		self.normB = LayerNorm2d(channels)
		self.proj1A = nn.Conv2d(channels, channels, 1)
		self.proj1B = nn.Conv2d(channels, channels, 1)
		self.beta  = nn.Parameter(torch.zeros((1,channels,1,1)), requires_grad=True)
		self.gamma = nn.Parameter(torch.zeros((1,channels,1,1)), requires_grad=True)
		self.proj2A = nn.Conv2d(channels, channels, 1)
		self.proj2B = nn.Conv2d(channels, channels, 1)

	def forward(self, x):
		xA, xB = x.chunk(2, dim=1)
		qA = self.proj1A(self.normA(xA)).permute(0,2,3,1)
		qB = self.proj1B(self.normB(xB)).permute(0,2,1,3)
		vA = self.proj2A(xA).permute(0,2,3,1)
		vB = self.proj2B(xB).permute(0,2,3,1)
		att = torch.matmul(qA, qB) * self.scale
		fA = torch.matmul(torch.softmax(att, dim=-1), vB)
		fB = torch.matmul(torch.softmax(att.permute(0,1,3,2), dim=-1), vA)
		fA = fA.permute(0,3,1,2) * self.beta
		fB = fB.permute(0,3,1,2) * self.gamma
		return torch.cat((xA + fA, xB + fB), dim=1)

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modules.common import *
from modules.reduce import *
from modules.attention import *


class AttentionConvertBlock(nn.Sequential):
	def __init__(self, channels_in, channels_out, kernel_size):
		super(AttentionConvertBlock,self).__init__(
			nn.Conv2d(
				channels_in, channels_out, kernel_size, 
				stride=1, padding=kernel_size//2),
			nn.LeakyReLU(),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			SpatialAttentionBlock(channels_out),
			nn.LeakyReLU())

class AttentionDownsampleBlock(nn.Sequential):
	def __init__(self, channels_in, channels_out, kernel_size):
		super(AttentionDownsampleBlock,self).__init__(
			Downsample2d(channels_out//channels_in, channels_in, channels_out, kernel_size),
			nn.LeakyReLU(),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			SpatialAttentionBlock(channels_out),
			nn.LeakyReLU())

class AttentionUpsampleBlock(nn.Sequential):
	def __init__(self, channels_in, channels_out, kernel_size):
		super(AttentionUpsampleBlock,self).__init__(
			Upsample2d(
				channels_in//channels_out, 
				channels_in, channels_out, 
				kernel_size),
			nn.LeakyReLU(),
			#nn.Conv2d(
			#	channels_in, channels_out, kernel_size, 
			#	stride=1, padding=kernel_size//2),
			#nn.LeakyReLU(),
			#nn.Conv2d(
			#	channels_out, channels_out * ((channels_in//channels_out) ** 2), 
			#	kernel_size, stride = 1, padding=kernel_size//2),
			#nn.BatchNorm2d(channels_out * ((channels_in//channels_out) ** 2)),
			#nn.PixelShuffle(channels_in//channels_out),
			#nn.PReLU(),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			SpatialAttentionBlock(channels_out),
			nn.LeakyReLU())


def initWeights(module):
	if isinstance(module, nn.Conv2d):
		assert 0 not in module.weight.shape
		nn.init.xavier_uniform_(module.weight)

class AttentionGenerator(nn.Module):
	def __init__(self, inputC, outputC, squeezeFilters = 32, expandFilters = 64, depth = 3):
		super(AttentionGenerator, self).__init__()
		
		sf = squeezeFilters
		self.da1 = AttentionConvertBlock    (inputC, sf, 3)
		self.da2 = AttentionDownsampleBlock (sf*1, sf*2, 3)
		self.da3 = AttentionDownsampleBlock (sf*2, sf*4, 3)
		self.da4 = AttentionUpsampleBlock   (sf*4, sf*2, 3)
		self.da5 = AttentionUpsampleBlock   (sf*2, sf*1, 3)
		self.convOut = nn.Conv2d(sf,outputC,1,)
		
		self.apply(initWeights)
		
		if not inputC == outputC:
			self.convIO = nn.Conv2d(inputC, outputC, 1, bias=False)
			self.convIO.weight = nn.Parameter(torch.ones([outputC, inputC, 1, 1]) / inputC)
			for p in self.convIO.parameters(): p.requires_grad = False
		else:
			self.convIO = None

	def forward(self, x):
		a = self.da1(x)
		b = self.da2(a)
		c = self.da3(b)
		d = self.da4(c) + b
		e = self.da5(d) + a

		if (self.convIO): 
			x = self.convIO(x) 
		
		y = self.convOut(e)
		y = torch.tanh(x + y)
		return y



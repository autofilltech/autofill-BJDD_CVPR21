import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.common import *
from modules.reduce import *
from modules.attention import CrossAttentionBlock

#https://github.com/megvii-research/NAFNet

class NAFBlock(nn.Module):
	def __init__(self, channels, dwExpand=2, ffnExpand=2):
		super(NAFBlock, self).__init__()
		dwChannels = channels * dwExpand
		self.conv1 = nn.Conv2d(channels, dwChannels, 1)
		self.conv2 = nn.Conv2d(dwChannels, dwChannels, 3, padding=1, groups = dwChannels)
		self.conv3 = nn.Conv2d(dwChannels // 2, channels, 1)

		self.sca = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(dwChannels // 2, dwChannels // 2, 1))
		
		self.gate = Gate()
		
		ffnChannels = channels * ffnExpand
		self.conv4 = nn.Conv2d(channels, ffnChannels, 1)
		self.conv5 = nn.Conv2d(ffnChannels // 2, channels, 1) 
		self.norm1 = LayerNorm2d(channels)
		self.norm2 = LayerNorm2d(channels)
		self.beta = nn.Parameter(torch.zeros((1,channels,1,1)), requires_grad=True)
		self.gamma = nn.Parameter(torch.zeros((1,channels,1,1)), requires_grad=True)

	def forward(self, x):
		orig = x
		x = self.norm1(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.gate(x)
		x = x * self.sca(x)
		x = self.conv3(x)
		y = orig + x * self.beta
		x = self.norm2(y)
		x = self.conv4(x)
		x = self.gate(x)
		x = self.conv5(x)
		return y + x * self.gamma

class NAFSRBlock(nn.Module):
	def __init__(self, channels, views=2):
		assert views == 2
		super(NAFSRBlock, self).__init__()
		self.block = NAFBlock(channels)
		self.fusion = CrossAttentionBlock(channels) # 2 views only
		self.views = views

	def forward(self, x):
		views = x.chunk(self.views, dim=1)
		features = torch.cat([self.block(v) for v in views], dim=1)
		features = self.fusion(features)
		return features


class NAFNet(nn.Module):
	def __init__(self, in_channels, width=32, num_encoders=[2,2,2,2], num_decoders=[2,2,2,2], num_extra=2):
		super(NAFNet,self).__init__()

		self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
		self.conv2 = nn.Conv2d(width, in_channels, 3, padding=1)

		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()
		self.extra = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()

		channels = width
		for num in num_encoders:
			self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
			self.downs.append(Downsample2d(2, channels, 2 * channels, 2))
			channels = channels * 2

		self.extra = nn.Sequential(*[NAFBlock(channels) for _ in range(num_extra)])

		for num in num_decoders:
			channels = channels // 2
			self.ups.append(Upsample2d(2, channels * 2, channels, 1))
			self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))

		self.padsize = 2 ** len(self.encoders)

	def forward(self, x):
		n,c,h,w = x.shape

		mod_pad_y = (self.padsize - h % self.padsize) % self.padsize
		mod_pad_x = (self.padsize - w % self.padsize) % self.padsize
		if mod_pad_y > 0 or mod_pad_x > 0:
			x = F.pad(x, (0, mod_pad_x, 0, mod_pad_y))

		orig = x
		x = self.conv1(x)
		encoders = []
		for encoder,down in zip(self.encoders, self.downs):
			x = encoder(x)
			encoders.append(x)
			x = down(x)

		x = self.extra(x)
		
		for decoder, up, skip in zip(self.decoders, self.ups, encoders[::-1]):
			x = up(x)
			x = x + skip
			x = decoder(x)

		x = self.conv2(x)
		x = x + orig

		return x[:,:,:h,:w]


class NAFSRNet(nn.Module):
	def __init__(self, scale_up, in_channels, views=2, width=48, num_blocks=16):
		super(NAFSRNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
		self.body = nn.Sequential(*[NAFSRBlock(width) for _ in range(num_blocks)])
		self.up = Upsample2d(2, width, in_channels, kernel_size=3)
		self.scale_up = scale_up
		self.views = views

	def forward(self, x):
		x_hr = F.interpolate(x, scale_factor=self.scale_up, mode='bilinear')
		views = x.chunk(self.views, dim=1)
		features = torch.cat([self.conv1(v) for v in views], dim=1)
		y = self.body(features)
		y = y.chunk(self.views, dim=1)
		y = torch.cat([self.up(v) for v in y], dim=1)
		y = y + x_hr
		return y

import torch
import torch.nn as nn
from modules.reduce import *

class SeparableConv2d(nn.Sequential):
	def __int__(self, features_in, features_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
		super(SeparableConv2D, self).__init__(
			nn.Conv2d(
				features_in, features_in, kernel_size, 
				stride, padding, dilation, 
				groups=channels_in, bias=bias),
			nn.Conv2d(
				features_in, features_out, 1,
				stride=1, padding=0, dilation=1,
				bias=bias))

class ResBlock(nn.Sequential):
	def __init__(self, channels, intermediate, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(ResBlock, self).__init__(
			ReduceAdd(
				Identity(),
				nn.Sequential(
					nn.Conv2d(
						channels_in, intermediate, kernel_size, 
						stride, padding, dilation, groups,
						bias = bias),
					nn.ReLU(),
					nn.Conv2d(
						intermediate, channels_in, kernel_size,
						stride, padding, dilation, groups,
						bias = bias))))

class View(nn.Module):
	def __init__(self, *args):
		super(View, self).__init__()
		self.shape = args
	
	def forward(self, x):
		return x.view(*self.shape)

class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args
	
	def forward(self, x):
		return x.reshape(*self.shape)

class Identity(nn.Module):
	def forward(self, x):
		return x

class PSUpsample2d(nn.Sequential):
	def __init__(self, scale, channels_in, channels_out, kernel_size, batchnorm=True, bias=False):
		super(PSUpsample2d, self).__init__(
			nn.Conv2d(channels_in, channels_out*(scale**2), kernel_size,
				stride=1, padding=kernel_size//2, bias=bias),
			nn.BatchNormalize(channels_out*(scale**2)) if batchnorm else Identity(),
			nn.PixelShuffle(scale))

class Downsample2d(nn.Conv2d):
	def __init__(self, scale, channels_in, channels_out, kernel_size, bias=False):
		super(Downsample2d, self).__init__(
			channels_in, channels_out, kernel_size,
			padding=kernel_size//2, stride=scale, bias=bias)

class Upsample2d(nn.ConvTranspose2d):
	def __init__(self, scale, channels_in, channels_out, kernel_size, bias=False):
		super(Upsample2d, self).__init__(
			channels_in, channels_out, kernel_size,
			stride=2, padding=kernel_size//2, 
			output_padding=1, bias=bias)

class PatchUnshuffle(nn.Module):
	def __init__(self, patches):
		super(PatchUnshuffle, self).__init__()
		self.patches = patches

	def forward(self,x):
		ny,nx = self.patches
		n,c,h,w = x.shape
		assert 0 == h % ny
		assert 0 == w % nx
		sy = h // ny
		sx = w // nx
		y = x.unfold(2, sy, sy).unfold(3, sx, sx)
		y = y.reshape(n, c * ny * nx, sy, sx)
		return y

class PatchShuffle(nn.Module):
	def __init__(self,patches):
		super(PatchShuffle, self).__init__()
		self.patches = patches

	def forward(self, x):
		ny,nx = self.patches
		n,c,h,w = x.shape
		sy = h * ny
		sx = w * nx
		nc = c // (ny * nx)
		y = x.view(n, nc, ny, nx, h, w)
		y = y.permute(0,1,2,4,3,5).reshape(n, nc, sy, sx)
		return y
		y = y.fold(3, w, w).fold(2, h, h)
		return y

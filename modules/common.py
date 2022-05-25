import torch
import torch.nn as nn
from modules.reduce import *


class Gate(nn.Module):
	def forward(self, x):
		x1, x2 = x.chunk(2, dim=1)
		return x1*x2

#https://github.com/megvii-research/NAFNet
class LayerNormFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, weight, bias, eps):
		ctx.eps = eps
		n,c,h,w = x.shape
		mu = x.mean(1, keepdim=True)
		var = (x-mu).pow(2).mean(1, keepdim=True)
		y = (x-mu) / (var + eps).sqrt()
		ctx.save_for_backward(y, var, weight)
		y = weight.view(1,c,1,1) * y + bias.view(1,c,1,1)
		return y

	@staticmethod
	def backward(ctx, grad_output):
		eps = ctx.eps
		n,c,h,w = grad_output.shape
		y,var,weight = ctx.saved_variables
		g = grad_output * weight.view(1,c,1,1)
		mu_g = g.mean(1, keepdim=True)
		mu_gy = (g * y).mean(1, keepdim=True)
		gx = 1. / torch.sqrt(var+eps) * (g - y * mu_gy - mu_g)
		#return gx, (grad_output * y).sum(dim=(3,2,0)), grad_output.sum(dim=(3,2,0)), None
		return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

#https://github.com/megvii-research/NAFNet
class LayerNorm2d(nn.Module):
	def __init__(self, channels, eps=1e-6):
		super(LayerNorm2d, self).__init__()
		self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
		self.eps = eps

	def forward(self, x):
		return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


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

class PixelShuffleUpsample2d(nn.Sequential):
	def __init__(self, scale, channels_in, channels_out, kernel_size, batch_norm=False, bias=False):
		super(PixelShuffleUpsample2d, self).__init__(
			nn.Conv2d(channels_in, channels_out*(scale**2), kernel_size,
				stride=1, padding=kernel_size//2, bias=bias),
			nn.BatchNormalize(channels_out*(scale**2)) if batch_norm else Identity(),
			nn.PixelShuffle(scale))

class Downsample2d(nn.Conv2d):
	def __init__(self, scale, channels_in, channels_out, kernel_size, batch_norm=False, bias=False):
		super(Downsample2d, self).__init__(
			channels_in, channels_out, kernel_size,
			padding=(kernel_size-1)//2, stride=scale, bias=bias)

class ConvTransposeUpsample2d(nn.ConvTranspose2d):
	def __init__(self, scale, channels_in, channels_out, kernel_size, bias=False):
		super(ConvTransposeUpsample2d, self).__init__(
			channels_in, channels_out, kernel_size,
			stride=scale, padding=kernel_size//2, 
			output_padding=1, bias=bias)

# set default upsample implementation to <base>
class Upsample2d(PixelShuffleUpsample2d):
	def __init__(self, scale, channels_in, channels_out, kernel_size, batch_norm=False, bias=False):
		super(Upsample2d,self).__init__(
				scale=scale, 
				channels_in=channels_in, 
				channels_out=channels_out, 
				kernel_size=kernel_size, 
				batch_norm=batch_norm,
				bias=bias)

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

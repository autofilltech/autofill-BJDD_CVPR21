import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common import *


def polarDefaultNormalize(x):
	n,c,h,w = x.shape
	assert 0 == (h % 4)
	assert 0 == (w % 4)
	assert 1 == c

	mean = torch.tensor([[[
		[0.485, 0.485, 0.486, 0.486],
		[0.485, 0.485, 0.486, 0.486],
		[0.486, 0.486, 0.406, 0.406],
		[0.486, 0.486, 0.406, 0.406]]]]).repeat(n, 1, h//4, w//4)
	std = torch.tensor([[[
		[0.229, 0.229, 0.224, 0.224],
		[0.229, 0.229, 0.224, 0.224],
		[0.224, 0.224, 0.224, 0.224],
		[0.224, 0.224, 0.224, 0.224]]]]).repeat(n, 1, h//4, w//4)

	return (x - mean.to(x.device)) / std.to(x.device)



class Stokes(nn.Module):
	RAW = 0
	YUV = 1
	RGB = 2
	LOSS = 3
	def __init__(self, color=None):
		super(Stokes,self).__init__()
		self.color = color
		if self.color is Stokes.RGB:
			self.convert = YuvToRgb()

	def forward(self, x):
		assert x.dim() == 4
		assert x.shape[1] == 4
		x = x ** 2
		e = 0.0001
		s0 = x.sum(dim=1, keepdim=True) / 2
		s1 = (chan(x,0) - chan(x,3))
		s2 = (chan(x,1) - chan(x,2))
		#if not self.color is Stokes.LOSS:
		s1 = s1 / (s0 + e)
		s2 = s2 / (s0 + e)
		y = torch.cat((s0,s1,s2), dim=1)
		if self.color is Stokes.LOSS:
			return y

		if self.color in (Stokes.YUV, Stokes.RGB):
			y[:,0,:,:] = (torch.sqrt(chan(y,1)**2+chan(y,2)**2) / 2).squeeze(1)
		if self.color is Stokes.RGB:
			y = self.convert(y)
		return y

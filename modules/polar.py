import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common import *


class Stokes(nn.Module):
	RAW = 0
	YUV = 1
	RGB = 2
	def __init__(self, color=None):
		super(Stokes,self).__init__()
		self.color = color
		if self.color is Stokes.RGB:
			self.convert = YuvToRgb()

	def forward(self, x):
		assert x.dim() == 4
		assert x.shape[1] == 4
		x = x ** 2
		e = 0.00001
		s0 = x.sum(dim=1, keepdim=True) / 2
		s1 = (chan(x,0) - chan(x,3)) / (s0 + e)
		s2 = (chan(x,1) - chan(x,2)) / (s0 + e)
		y = torch.cat((s0,s1,s2), dim=1)

		if self.color in (Stokes.YUV, Stokes.RGB):
			y[:,0,:,:] = (torch.sqrt(chan(y,1)**2+chan(y,2)**2) / 2).squeeze(1)
		if self.color is Stokes.RGB:
			y = self.convert(y)
		return y

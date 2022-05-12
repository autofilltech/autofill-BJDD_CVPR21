import torch
import torch.nn as nn

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

		


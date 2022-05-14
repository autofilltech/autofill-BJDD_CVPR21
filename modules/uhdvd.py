import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common import *
from modules.reduce import *
from modules.attention import *

class UHDVDRCSABlock(nn.Sequential):
	def __init__(self, channels_in, channels_out, shuffle, kernel_size=3, stride=1, padding=1, bias=False):
		super(UHDVDRCSABlock, self).__init__(
			nn.Conv2d(
				channels_in, channels_out, kernel_size, 
				padding=padding, stride=stride,	bias=bias),
			RCSABlock(channels_out))

class UHDVDResBlock(nn.Sequential):
	def __init__(self, channels_in, channels_out, shuffle, kernel_size=4, stride=1, padding=1, bias=False):
		super(UHDVDResBlock, self).__init__(
			ResBlock(channels_in, channels_in, kernel_size, padding=kernel_size//2)
			nn.ConvTranspose2d(
				channels_in, channels_out, kernel_size, 
				padding = padding, stride = stride, bias = bias),

class UHDVDEncoder(nn.Sequential):
	def __init__(self, scale, channels_in, shuffle):
		mul = shuffle[0] * shuffle[1]
		super(UHDVDEncoder, self).__init__(
			UHDVDRCSABlock(channels_in, 32, shuffle),
			UHDVDRCSABlock(32, 64, shuffle, 3, stride=2),
			UHDVDRCSABlock(64, 128, shuffle, 3, stride=2),

class UHDVDDecoder(nn.Sequential):
	def __init__(self, scale, channels_out, shuffle):
		super(UHDVDDecoder, self).__init__(
			UHDVDResBlock(128, 64, shuffle, stride=2),
			UHDVDResBlock(64, 32, shuffle, stride=4),
			ResBlock(32, 32, 3, padding=1),
			nn.Conv2d(32, channels_out, 3, stride=1, padding=1, groups=mul),

class UHDVD(nn.Module):
	def __init__(self):
		super(UHDVD, self).__init__()
		self.enc1 = UHDVDEncoder(1, 6, (1,1))
		self.enc2 = UHDVDEncoder(2, 6, (1,2))
		self.enc4 = UHDVDEncoder(4, 6, (2,2))
		self.enc8 = UHDVDEncoder(8, 6, (2,4))

		self.dec1 = UHDVDDecoder(1, 3, (1,1))
		self.dec2 = UHDVDDecoder(2, 3, (1,1))
		self.dec4 = UHDVDDecoder(4, 3, (1,2))
		self.dec8 = UHDVDDecoder(8, 3, (2,2))

		self.up8 = Upscale2d(2, 3, 3, 4);
		self.up4 = Upscale2d(2, 3, 3, 4);
		self.up2 = Upscale2d(2, 3, 3, 4);

	def forward(self, x):
		# cat x with previous frame
		# move through pipelines
		x1 = x
		x2 = F.avg_pool2d(x1, 2)
		x4 = F.avg_pool2d(x2, 2)
		x8 = F.avg_pool2d(x4, 2)

		i8 = self.enc8(x8);
		y8 = self.dec8(i8);

		i4 = self.enc4(x4 + y8) + self.up8(y8)
		y4 = self.dec4(i4)

		i2 = self.enc2(x2 + y4) + self.up4(y4)
		y2 = self.dec2(i2)

		i1 = self.enc1(x1 + y2) + self.up2(y2)
		y1 = self.dec1(i1)

		y = y1
		return y
		

		
		
		


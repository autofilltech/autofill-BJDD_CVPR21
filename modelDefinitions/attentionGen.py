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
				stride=1, padding=(kernel_size-1)//2),
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
			#Upsample2d(
			#	channels_in//channels_out, 
			#	channels_in, channels_out, 
			#	kernel_size),
			#nn.LeakyReLU(),
			nn.Conv2d(
				channels_in, channels_out, kernel_size, 
				stride=1, padding=(kernel_size-1)//2),
			nn.LeakyReLU(),
			nn.Conv2d(
				channels_out, channels_out * ((channels_in//channels_out) ** 2), 
				kernel_size, stride = 1, padding=(kernel_size-1)//2),
			nn.BatchNorm2d(channels_out * ((channels_in//channels_out) ** 2)),
			nn.PixelShuffle(channels_in//channels_out),
			nn.PReLU(),
			#nn.Conv2d(channels_out, channels_out, kernel_size, padding=(kernel_size-1)//2),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			AttentionResBlock(channels_out),
			SpatialAttentionBlock(channels_out),
			nn.LeakyReLU())


def initWeights(module):
	if isinstance(module, nn.Conv2d):
		assert 0 not in module.weight.shape
		nn.init.xavier_uniform_(module.weight)
		if not module.bias is None: module.bias.data.zero_()

class AttentionGenerator(nn.Module):
	def __init__(self, inputC, outputC, squeezeFilters = 32, depth = 3):
		super(AttentionGenerator, self).__init__()
		
		sf = squeezeFilters
		self.da1 = AttentionConvertBlock    (inputC, sf, 7)
		self.da2 = AttentionDownsampleBlock (sf*1, sf*2, 3)
		self.da3 = AttentionDownsampleBlock (sf*2, sf*4, 3)
		self.da4 = AttentionDownsampleBlock (sf*4, sf*8, 3)
		self.da5 = AttentionUpsampleBlock   (sf*8, sf*4, 3)
		self.da6 = AttentionUpsampleBlock   (sf*4, sf*2, 3)
		self.da7 = AttentionUpsampleBlock   (sf*2, sf*1, 3)
		self.convOut = nn.Sequential(
				#nn.Conv2d(sf,sf,3, padding=1),
				#nn.PReLU(),
				nn.Conv2d(sf,outputC,1,padding=0))
		
		self.apply(initWeights)
		
		if not inputC == outputC:
			# this works for 4x4 pixel block
			kernel = torch.tensor([[1,2,2,2,1],[2,4,4,4,2],[2,4,4,4,2],[2,4,4,4,2],[1,2,2,2,1]]) / 64
			self.convIO = nn.Conv2d(inputC, outputC, 1, bias=False, padding=0)
			self.convIO.weight = nn.Parameter(torch.ones([outputC, inputC, 1, 1]) / inputC)
			#self.convIO = nn.Conv2d(inputC, outputC, 1, bias=False, padding=2)
			#self.convIO.weight = nn.Parameter(kernel.expand([outputC, inputC, 5, 5]))
			for p in self.convIO.parameters(): p.requires_grad = False
		else:
			self.convIO = None

	def forward(self, x):
		n,c,h,w = x.shape
		a = self.da1(x)
		b = self.da2(a)
		c = self.da3(b)
		d = self.da4(c)
		e = self.da5(d) + c
		f = self.da6(e) + b
		g = self.da7(f) + a

		if (self.convIO): 
			x = self.convIO(x) 

		#mo = torch.tensor([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]).to(x.device)
		#mg = torch.tensor([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]).to(x.device)
		#x[:,0,:,:] *= mo.roll(0,0).repeat(n,h//4,w//4)
		#x[:,1,:,:] *= mg.roll(0,-2).repeat(n,h//4,w//4)
		#x[:,2,:,:] *= mo.roll(-2,-2).repeat(n,h//4,w//4)	
		#x[:,3,:,:] *= mo.roll(0,1).repeat(n,h//4,w//4)
		#x[:,4,:,:] *= mg.roll(0,-1).repeat(n,h//4,w//4)
		#x[:,5,:,:] *= mo.roll(-2,-1).repeat(n,h//4,w//4)
		#x[:,6,:,:] *= mo.roll(1,0).repeat(n,h//4,w//4)
		#x[:,7,:,:] *= mg.roll(1,-2).repeat(n,h//4,w//4)
		#x[:,8,:,:] *= mo.roll(-1,-2).repeat(n,h//4,w//4)	
		#x[:,9,:,:] *= mo.roll(1,1).repeat(n,h//4,w//4)
		#x[:,10,:,:] *= mg.roll(1,-1).repeat(n,h//4,w//4)
		#x[:,11,:,:] *= mo.roll(-1,-1).repeat(n,h//4,w//4)

		#x = F.pixel_unshuffle(x,2)
		#x = F.pixel_unshuffle(x,2)
		#x = F.interpolate(x, scale_factor=4, mode="bilinear")
		#x = (x[:,[0,1,3,4,5,7,8,9,11,12,13,15],:,:] + x[:,[0,2,3,4,6,7,8,10,11,12,14,15],:,:]) / 2

		y = self.convOut(g)
		#y = F.tanh(x+y)
		y = x + y
		return y



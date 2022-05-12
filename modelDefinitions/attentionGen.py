import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *	
#from modules.common import *
#from modules.attention import *
class attentionNet(nn.Module):
	def __init__(self, inputC, outputC, squeezeFilters = 8, expandFilters = 16, depth = 3):
		super(attentionNet, self).__init__()
		k = 3
		# Input Block
		self.inputConv = nn.Conv2d(inputC, squeezeFilters, 7, stride=1, padding=7//2)

		depthAttenBlock = []
		for i in range (depth):
			depthAttenBlock.append(attentionGuidedResBlock(squeezeFilters, expandFilters))
		self.depthAttention1 = nn.Sequential(*depthAttenBlock)
		self.spatialAttention1 = SpatialAttentionBlock(squeezeFilters)
		self.down1 = nn.Conv2d(squeezeFilters, squeezeFilters*2, k, 2, k//2) 

		depthAttenBlock1 = []
		for i in range (depth):
			depthAttenBlock1.append(attentionGuidedResBlock(squeezeFilters * 2, expandFilters * 2, dilationRate=1))
		self.depthAttention2 = nn.Sequential(*depthAttenBlock1)
		self.spatialAttention2 = SpatialAttentionBlock(squeezeFilters * 2)
		self.down2 = nn.Conv2d(squeezeFilters*2, squeezeFilters*4, k, 2, k//2) 

		depthAttenBlock3 = []
		for i in range (depth):
			depthAttenBlock3.append(attentionGuidedResBlock(squeezeFilters * 4, expandFilters * 4, dilationRate=1))
		self.depthAttention3 = nn.Sequential(*depthAttenBlock3)
		self.spatialAttention3 = SpatialAttentionBlock(squeezeFilters * 4)
		self.convUP1 = nn.Conv2d(squeezeFilters * 4, squeezeFilters *2, k, 1, k//2) 
		self.psUpsampling1 = pixelShuffleUpsampling(inputFilters=squeezeFilters * 2, scailingFactor=2)

		depthAttenBlock4 = []
		for i in range (depth):
			depthAttenBlock4.append(attentionGuidedResBlock(squeezeFilters * 2, expandFilters * 2, dilationRate=1))
		self.depthAttention4 = nn.Sequential(*depthAttenBlock4)
		self.spatialAttention4 = SpatialAttentionBlock(squeezeFilters * 2)
		self.convUP2 = nn.Conv2d(squeezeFilters * 2, squeezeFilters, k, 1, k//2) 
		self.psUpsampling2 = pixelShuffleUpsampling(inputFilters=squeezeFilters, scailingFactor=2)

		# Output Block
		depthAttenBlock5 = []
		for i in range (depth):
			depthAttenBlock5.append(attentionGuidedResBlock(squeezeFilters,expandFilters, dilationRate=1))
		self.depthAttention5 = nn.Sequential(*depthAttenBlock5)
		self.spatialAttention5 = SpatialAttentionBlock(squeezeFilters)
		self.convOut = nn.Conv2d(squeezeFilters,outputC,1,)

		if not inputC == outputC:
			self.convIO = nn.Conv2d(inputC, outputC, 1, bias=False)
			self.convIO.weight = nn.Parameter(torch.ones([outputC, inputC, 1, 1]) / inputC)
			for p in self.convIO.parameters(): p.requires_grad = False
		else:
			self.convIO = None
		# Weight Initialization
		self._initialize_weights()

	def forward(self, img):
		#n,c,h,w = img.shape
		# img.expand((n,16,h,w))
		#x, mean = torch.split(self.norm(img), 1, dim=-3)
		#mean = self.upsample(mean.reshape(n,1,h,w))
		#img = x.reshape(n,1,h,w)
		#img = img * 2 - 1
		#img = F.pixel_unshuffle(img,2)
		#img = F.pixel_unshuffle(img,2)
		#mean = 0.5 #torch.mean(img)
		#sdev = 0.5 #torch.mean((img - mean)**2)**0.5
		#img = (img - mean) / sdev
		#img = F.pixel_shuffle(img, 2)
		#img = F.pixel_shuffle(img, 2)

		xInp = F.leaky_relu(self.inputConv(img))
		xSP1 = self.depthAttention1(xInp)
		xFA1 = F.leaky_relu(self.spatialAttention1(xSP1))

		xDS1 = F.leaky_relu(self.down1(xFA1))
		xSP2 = self.depthAttention2(xDS1)
		xFA2 = self.spatialAttention2(xSP2) 

		xDS2 = F.leaky_relu(self.down2(xFA2))
		xSP3 = self.depthAttention3(xDS2)
		xFA3 = self.spatialAttention3(xSP3)

		xCP1 = F.leaky_relu(self.convUP1(xFA3))
		xPS1 = self.psUpsampling1(xCP1) 
		xSP4 = self.depthAttention4(xPS1)
		xFA4 = self.spatialAttention4(xSP4) + xFA2

		xCP2 = F.leaky_relu(self.convUP2(xFA4))
		xPS2 = self.psUpsampling2(xCP2) 
		xSP5 = self.depthAttention5(xPS2)
		xFA5 = self.spatialAttention5(xSP5) + xFA1
		 
		io = self.convIO(img) if self.convIO else img
		#out = self.convOut(xFA5)
		#out = torch.cat((out, mean), dim=1)
		#return torch.tanh(self.denorm(out))
		img = torch.tanh(self.convOut(xFA5) + io)
		#img = img * sdev + mean
		return img

	def _initialize_weights(self):

		self.inputConv.apply(init_weights)
		self.depthAttention1.apply(init_weights)
		self.spatialAttention1.apply(init_weights)
		
		self.down1.apply(init_weights)
		self.depthAttention2.apply(init_weights)
		self.spatialAttention2.apply(init_weights)
		
		self.down2.apply(init_weights)
		self.depthAttention3.apply(init_weights)
		self.spatialAttention3.apply(init_weights)
		
		self.convUP1.apply(init_weights)
		self.psUpsampling1.apply(init_weights)
		self.depthAttention4.apply(init_weights)
		self.spatialAttention4.apply(init_weights)
	   
		self.convUP2.apply(init_weights)
		self.psUpsampling2.apply(init_weights)
		self.depthAttention5.apply(init_weights)
		self.spatialAttention5.apply(init_weights)
		
		self.convOut.apply(init_weights)

#net = attentionNet()
#summary(net, input_size = (3, 128, 128))
#print ("reconstruction network")

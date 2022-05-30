import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19
from skimage import color
import random
import numpy as np
from modules.common import RgbToYuv

class FeatureLoss(nn.Module):
	def __init__(self, regulator = 1.0):
		super(FeatureLoss, self).__init__()
		vgg19_model = vgg19(pretrained=True)
		self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
		self.loss = torch.nn.L1Loss()
		self.regulator = regulator

	def forward(self, x, y):
		size = x.size()
		n,c,h,w = x.shape
		self.feature_extractor = self.feature_extractor.to(x.device)
		self.loss = self.loss.to(x.device)
		
		#i = [random.randrange(0, c//3)]
		i = range(c//3)
		featureLoss = 0
		for idx in i:
			# VGG Feature Loss
			x0 = self.feature_extractor(x[:,idx*3:idx*3+3,:,:] * 2 - 1)
			y0 = self.feature_extractor(y[:,idx*3:idx*3+3,:,:] * 2 - 1)
			featureLoss += self.loss(x0, y0) * self.regulator
		
		# TV loss
		dh = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1:, :] - y[:, :, :-1, :]))).sum()
		dw = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :, 1:] - y[:, :, :, :-1]))).sum()
		tvLoss = (dh + dw) / size[0] / size[1] / size[2] / size[3]
		
		# Total Loss
		totalLoss = featureLoss * tvLoss
		return totalLoss


class YuvLoss(nn.Module):
	def __init__(self):
		super(YuvLoss, self).__init__()
		self.yuv = RgbToYuv()

	def forward(self, x, y):
		a = self.yuv(x.clamp(0,1))
		b = self.yuv(y.clamp(0,1))
		assert not torch.isnan(a).any()
		assert not torch.isnan(b).any()
		d = (a - b) ** 2
		d = (d[:,0,:,:] + d[:,1,:,:] + d[:,2,:,:])
		return d.mean()

# I don't trust this module... TODO rewrite using torch instead of scikit
class ColorLoss(nn.Module):
	def __init__(self):
		super(ColorLoss, self).__init__()
		
	def forward(self, x, y):
		loss = []
		n,c,h,w = x.shape

		for b in range(n):
			#i = [random.randrange(0, c//3)]
			i = range(c//3)
			for idx in i:
				x0 = x[b, idx*3:idx*3+3, :, :]
				y0 = y[b, idx*3:idx*3+3, :, :]
				assert x0.dim() == 3
				assert y0.dim() == 3
				ximg = torch.clamp(x0,0,1).permute(1,2,0).detach().cpu().numpy()
				yimg = torch.clamp(y0,0,1).permute(1,2,0).detach().cpu().numpy()
				delta = np.absolute(color.deltaE_ciede2000(color.rgb2lab(ximg), color.rgb2lab(yimg))) / 100.0
				loss.append(np.mean(delta))

		loss = torch.mean(torch.tensor(loss, requires_grad=True).to(x.device))
		return loss


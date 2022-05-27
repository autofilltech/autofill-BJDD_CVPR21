import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19
import random

class PerceptualLoss(nn.Module):
	def __init__(self, regulator = 1.0):
		super(PerceptualLoss, self).__init__()
		vgg19_model = vgg19(pretrained=True)
		self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
		self.loss = torch.nn.L1Loss()
		self.regulator = regulator

	def forward(self, x, y):
		size = x.size()
		self.feature_extractor = self.feature_extractor.to(x.device)
		self.loss = self.loss.to(x.device)

		# VGG Feature Loss
		x0 = self.feature_extractor(x[:,0:3,:,:])
		x1 = self.feature_extractor(x[:,3:6,:,:])
		y0 = self.feature_extractor(y[:,0:3,:,:])
		y1 = self.feature_extractor(y[:,3:6,:,:])
		featureLoss = (self.loss(x0, y0) + self.loss(x1, y1)) * self.regulator

		# TV loss
		dh = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1:, :] - y[:, :, :-1, :]))).sum()
		dw = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :, 1:] - y[:, :, :, :-1]))).sum()
		tvLoss = (dh + dw) / size[0] / size[1] / size[2] / size[3]
		
		# Total Loss
		totalLoss = featureLoss * tvLoss
		return totalLoss

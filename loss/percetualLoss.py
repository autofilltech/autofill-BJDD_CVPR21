import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19
import random

class regularizedFeatureLoss(nn.Module):
	def __init__(self, device, percepRegulator = 1.0):
		super(regularizedFeatureLoss, self).__init__()
		vgg19_model = vgg19(pretrained=True)
		self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
		self.loss = torch.nn.L1Loss()
		self.percepRegulator = percepRegulator
		self.device = device

	def forward(self, x, y):
		i = random.randrange(0, x.shape[1]-3)
		# VGG Feature Loss
		genFeature = self.feature_extractor(x[:,i:i+3,:,:])
		gtFeature = self.feature_extractor(y[:,i:i+3,:,:])
		featureLoss = self.loss(genFeature, gtFeature) * self.percepRegulator

		# TV loss
		size = x.size()
		h_tv_diff = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1:, :] - y[:, :, :-1, :]))).sum()
		w_tv_diff = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :, 1:] - y[:, :, :, :-1]))).sum()
		tvloss = (h_tv_diff + w_tv_diff) / size[0] / size[1] / size[2] / size[3]

		# Total Loss
		totalLoss = tvloss * featureLoss 
		return totalLoss

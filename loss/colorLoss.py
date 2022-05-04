import torch
import torch.nn as nn
from skimage import io, color
import numpy as np
import random
class deltaEColorLoss(nn.Module):

	def __init__(self, normalize=None):
		super(deltaEColorLoss, self).__init__()
		self.loss = []
		self.normalize = normalize
		self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	
	def torchTensorToNumpy(self, image):
		imageNP = image.cpu().detach().numpy().reshape(image.shape[1], image.shape[2], image.shape[0])
		return imageNP

	def __call__(self, genImage, gtImage):
		self.loss.clear()
		
		for pair in range(len(genImage)):
			i = random.randrange(0, gtImage.shape[1]//3)
			gt = gtImage[pair][i*3:(i*3+3)]
			gen = genImage[pair][i*3:(i*3+3)]
			# Converting and changing shape of torch tensor into numpy
			imageGTNP = self.torchTensorToNumpy(gt)
			imageGenNP = self.torchTensorToNumpy(gen)

			# Calculating color difference
			deltaE = np.absolute(color.deltaE_ciede2000(color.rgb2lab(imageGTNP), color.rgb2lab(imageGenNP)))
			if self.normalize:
				deltaE /= 255.0

			# Mean deifference for an image pair
			self.loss.append(np.mean(deltaE))
		deltaELoss = torch.mean(torch.tensor(self.loss, requires_grad=True)).to(self.device)
		
		return deltaELoss

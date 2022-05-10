import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ICColorLoss(nn.Module):
	def __init__(self):
		super(ICColorLoss, self).__init__()
		pass

	def __call__(self, y): # y = [B, 12, H, W]
		assert len(y.shape) == 4
		assert y.shape[1] == 12
		assert y.dtype == torch.float32
		y = y ** 2
		# S0 = Luminance per color channel
		S0a = torch.stack((
			y[:,0,:,:] + y[:,9,:,:],
			y[:,1,:,:] + y[:,10,:,:],
			y[:,2,:,:] + y[:,11,:,:]),1) / 2
		S0b = torch.stack((
			y[:,3,:,:] + y[:,6,:,:],
			y[:,4,:,:] + y[:,7,:,:],
			y[:,5,:,:] + y[:,8,:,:]),1) / 2

		S0a = S0a ** (1.0/2.4)
		S0b = S0b ** (1.0/2.4)
		
		return torch.mean(torch.sqrt((S0a - S0b) ** 2))

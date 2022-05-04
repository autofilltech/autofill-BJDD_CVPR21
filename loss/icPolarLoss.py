import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ICPolarLoss(nn.Module):
	def __init__(self):
		super(ICPolarLoss, self).__init__()
		pass

	def toStokes(self, y):
		y = y ** 2

		S0 = torch.stack((
			y[:,0,:,:] + y[:,3,:,:] + y[:,6,:,:] + y[:,9,:,:],
			y[:,1,:,:] + y[:,4,:,:] + y[:,7,:,:] + y[:,10,:,:],
			y[:,2,:,:] + y[:,5,:,:] + y[:,8,:,:] + y[:,11,:,:]), 1)
		S0 /= 2
		print(S0.shape)		
		e = 0.0001
		S0 += e

		# S1 = Tl - BR
		S1 = torch.stack((
			y[:,0,:,:] - y[:,9,:,:],
			y[:,1,:,:] - y[:,10,:,:],
			y[:,2,:,:] - y[:,11,:,:]),1)
		S1 /= S0
		print(S1.shape)
		# S2 = TR - BL
		S2 = torch.stack((
			y[:,3,:,:] - y[:,6,:,:],
			y[:,4,:,:] - y[:,7,:,:],
			y[:,5,:,:] - y[:,8,:,:]),1)
		S2 /= S0
		print(S2.shape)
		return torch.stack((S0, S1, S2),1).squeeze(0).permute(1,0,2,3)

	def stokesToYuv(self, x):
		x[:,0,:,:] = torch.sqrt(x[:,1,:,:] ** 2 + x[:,2,:,:] ** 2) * 0.5
		return x

	def yuvToRgb(self, x):
		y = x[:,0,:,:]
		u = x[:,1,:,:] * 0.436
		v = x[:,2,:,:] * 0.615

		r = (y + v * 1.13983) * 2 - 1
		g = (y - u * 0.39465 - v * 0.58060) * 2 - 1
		b = (y + u * 2.03211) * 2 - 1

		return torch.stack((r,g,b), 1).clamp(-1,1)

	def __call__(self, y): # y = [B, 12, H, W]
		assert y.dtype == torch.float32
		y = y ** 2
		# S0 = Luminance per color channel
		S0 = torch.stack((
			y[:,0,:,:] + y[:,3,:,:] + y[:,6,:,:] + y[:,9,:,:],
			y[:,1,:,:] + y[:,4,:,:] + y[:,7,:,:] + y[:,10,:,:],
			y[:,2,:,:] + y[:,5,:,:] + y[:,8,:,:] + y[:,11,:,:]), 1)
		S0 /= 2

		# S1 = Tl - BR
		S1 = torch.stack((
			y[:,0,:,:] - y[:,9,:,:],
			y[:,1,:,:] - y[:,10,:,:],
			y[:,2,:,:] - y[:,11,:,:]),1)
		S1 /= S0

		# S2 = TR - BL
		S2 = torch.stack((
			y[:,3,:,:] - y[:,6,:,:],
			y[:,4,:,:] - y[:,7,:,:],
			y[:,5,:,:] - y[:,8,:,:]),1)
		S2 /= S0

		lossS1 = S1 - torch.mean(S1, 1, keepdim = True)
		lossS2 = S2 - torch.mean(S2, 1, keepdim = True)
		return torch.mean(torch.sqrt(lossS1 ** 2 + lossS2 ** 2))

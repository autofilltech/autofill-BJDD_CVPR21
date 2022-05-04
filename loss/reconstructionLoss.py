import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ReconstructionLoss(nn.Module):
	def __init__(self):
		pass

	def __call__(self, x, y): # x = [B, 1, H, W] y = [B, 12, H, W]
		assert x.shape[1] == 1
		assert y.shape[1] == 12

		
		rr = torch.stack((y[:,0,0::4,0::4], y[:,3,0::4, 1::4], y[:,6,1::4,0::4], y[:,9,1::4,1::4]), 1)
		gr = torch.stack((y[:,1,0::4,2::4], y[:,4,0::4, 3::4], y[:,7,1::4,2::4], y[:,10,1::4,3::4]), 1)
		gb = torch.stack((y[:,1,2::4,0::4], y[:,4,2::4, 1::4], y[:,8,3::4,0::4], y[:,10,3::4,1::4]), 1)
		bb = torch.stack((y[:,2,2::4,2::4], y[:,5,2::4, 3::4], y[:,9,3::4,2::4], y[:,11,3::4,3::4]), 1)
		predRaw = torch.concat((rr,gr,gb,bb),1)

		predRaw = nn.PixelShuffle(4)(torch.stack((
			predRaw[:,0,:,:], predRaw[:,1,:,:], predRaw[:,4,:,:], predRaw[:,5,:,:],
			predRaw[:,2], predRaw[:,3,:,:], predRaw[:,6,:,:], predRaw[:,7,:,:],
			predRaw[:,8], predRaw[:,9,:,:], predRaw[:,12,:,:], predRaw[:,13,:,:],
			predRaw[:,10], predRaw[:,11,:,:], predRaw[:,14,:,:], predRaw[:,5,:,:]),1))

		loss = torch.mean(torch.sqrt((predRaw - x) ** 2))
		return loss

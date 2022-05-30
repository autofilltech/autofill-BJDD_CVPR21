import os
from   glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as Tf
from   .extensions import AFCapture
import random
import cv2
import numpy as np

class B12Dataset(Dataset):
	def __init__(self, path, size=224, length=None, rotate=True, scale=True):
		super(B12Dataset, self).__init__()

		self.length = length
		self.targetPath = path
		self.ids = [os.path.split(d)[1] for d in glob(os.path.join(self.targetPath, "*"))]
		if type(size) is int: self.size = (size, size)
		else: self.size = size

		
		t = [T.GaussianBlur(kernel_size = 3, sigma = 1)]
		if rotate:
			t.append(T.RandomRotation(degrees = (-180, +180)))
		if scale:
			t.append(T.RandomResizedCrop(size=self.size, scale=(0.05, 0.25), ratio=(1,1)))
		else:
			t.append(T.RandomCrop(size=self.size))

		self.transforms = T.Compose(t)

	def __len__(self):
		return self.length if not self.length and self.length > len(self.ids) is None else len(self.ids)
	
	def __getitem__(self, idx):
		path = self.ids[idx]
		path = os.path.join(self.targetPath, path)

		if path.endswith(".b12"):
			b12 = AFCapture.readB12(path)
			assert len(b12.shape) == 3
		
		elif path.endswith(".pgm"):
			b12 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			assert len(b12.shape) == 2
			b12 = torch.Tensor(np.asarray(b12[:,:])/255.0).cuda().view(1,b12.shape[0], b12.shape[1])

		c,h,w = b12.shape
		ps = F.pixel_unshuffle(b12,2)
		ps = F.pixel_unshuffle(ps,2)
		ps = ps.reshape(4,4,h//4,w//4).permute(1,0,2,3)
		sr,mr = torch.std_mean(ps[0])
		sb,mb = torch.std_mean(ps[3])
		sg,mg = torch.std_mean(ps[1:2])
		ps[0] = (ps[0] - mr) / sr
		ps[1] = (ps[1] - mg) / sg
		ps[2] = (ps[2] - mg) / sg
		ps[3] = (ps[3] - mb) / sb
		ps = torch.clamp(ps, -1, 1)
		ps = ps.permute(1,0,2,3).reshape(16,h//4,w//4)
		# [RGGB RGGB RGGB RGGB]
		# b12 = F.pixel_shuffle(ps, 2)
		# b12 = F.pixel_shuffle(b12, 2)
		channelsG1 = [0,1,3,4,5,7,8, 9,11,12,13,15]
		channelsG2 = [0,2,3,4,6,7,8,10,11,12,14,15]
		ps = (ps[channelsG1,:,:] + ps[channelsG2,:,:])/2
		# [RGB RGB RGB RGB]

		assert ps.shape[0] == 12
		assert ps.shape[1] == b12.shape[1] // 4
		assert ps.shape[2] == b12.shape[2] // 4
		
		img = self.transforms(ps)
		raw = torch.stack((
			img[ 0, 0::4, 0::4],
			img[ 1, 0::4, 2::4],
			img[ 1, 2::4, 0::4],
			img[ 2, 2::4, 2::4],
			img[ 3, 0::4, 1::4], 
			img[ 4, 0::4, 3::4],
			img[ 4, 2::4, 1::4],
			img[ 5, 2::4, 3::4],
			img[ 6, 1::4, 0::4],
			img[ 7, 1::4, 2::4],
			img[ 7, 3::4, 0::4],
			img[ 8, 3::4, 2::4],
			img[ 9, 1::4, 1::4],
			img[10, 1::4, 3::4],
			img[10, 3::4, 1::4],
			img[11, 3::4, 3::4]))
			
		raw = F.pixel_shuffle(raw,2)
		raw = F.pixel_shuffle(raw,2)
		#raw = raw * 2 - 1
		#img = img * 2 - 1
		return raw, img

import os
from   glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as Tf
import torchvision.io as io

''' 
Stereo Super Resolution DataSet
Expects directory structure to be:
	<path>/target/<pair_id>/hr[0|1].png
'''
class SSRDataset(Dataset):
	def __init__(self, path, size=None, length=None):
		super(SSRDataset,self).__init__()
		self.length = length
		self.targetPath = path
		self.ids = [os.path.split(d)[1] for d in glob(os.path.join(self.targetPath, "*"))]
		if type(size) is int: self.size = (size, size)
		else: self.size = size

	def __len__(self):
		return self.length if not self.length is None else len(self.ids)

	def __getitem__(self, idx):
		idx = idx % len(self.ids)
		hrPath = os.path.join(self.targetPath, self.ids[idx])
		hr0 = io.read_image(os.path.join(hrPath, "hr0.png"))
		hr1 = io.read_image(os.path.join(hrPath, "hr1.png"))
		
		_,h,w = hr0.shape
		if self.size is None: 
			assert h % 2 == 0 and w % 2 == 0
			self.size = (h//2,w//2)
		
		marginX = w - self.size[1] * 2
		marginY = h - self.size[0] * 2
		offsetX = torch.randint(0, marginX // 4, (1,)).item() * 4
		offsetY = torch.randint(0, marginY // 4, (1,)).item() * 4
		
		hr0 = Tf.crop(hr0, offsetY, offsetX, self.size[0]*2, self.size[1]*2)
		hr1 = Tf.crop(hr1, offsetY, offsetX, self.size[0]*2, self.size[1]*2)
		
		hr = torch.cat((hr0, hr1), dim=-3) / 255.0
		lr = F.avg_pool2d(hr, 2, stride=2)

		return (lr, hr)

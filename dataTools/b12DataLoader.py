import glob
import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
from .extensions import AFCapture
import random
class b12DatasetReader(Dataset):
	def __init__(self, image_list, imagePathGT, height, width, transformation=True):
		self.image_list = image_list
		self.imagePathGT = imagePathGT
		self.imageH = height
		self.imageW = width
		self.normalize1 = T.Normalize(normMean(1), normStd(1))
		self.normalize12 = T.Normalize(normMean(16), normStd(16))
		self.all = False

		self.transformGT = T.Compose([
			T.GaussianBlur(kernel_size = 5, sigma = 1), 
			T.RandomRotation(degrees = (-180, +180)),
			T.RandomResizedCrop(size=244, scale=(0.25, 1), ratio=(1,1))
			])
		self.transformRI = T.Compose([AddGaussianNoise(0, 0.02, pov=1)])

	def __len__(self):
		return (len(self.image_list))
	
	def __getitem__(self, i):
		# Read Images
		#print ("print i",i, i+1)
		b12 = AFCapture.readB12(self.image_list[i])
		assert len(b12.shape) == 3
		c,h,w = b12.shape
		ps = F.pixel_unshuffle(b12,2)
		ps = F.pixel_unshuffle(ps,2)
		ps = ps.reshape(4,4,h//4,w//4).permute(1,0,2,3)
		ps[0] *= 2
		ps[3] *= 3
		ps = ps.permute(1,0,2,3).reshape(16,h//4,w//4)
		# [RGGB RGGB RGGB RGGB]
		b12 = F.pixel_shuffle(ps, 2)
		b12 = F.pixel_shuffle(b12, 2)
		# z = ps.reshape(4,4,ps.shape[1], ps.shape[2]).permute(1,0,2,3)
		# [RRRR, GGGG, GGGG, BBBB]
		#mean = torch.mean(z, dim=(1,2,3), keepdim=True)
		#mean[1] = (mean[1] + mean[2])/2
		#mean[2] = mean[1]
		#sdev = torch.mean((z - mean)**2, dim=(1,2,3), keepdim=True)**0.5
		#sdev[1] = (sdev[1] + sdev[2])/2
		#sdev[2] = sdev[1]
		#z = (z - mean) / sdev
		#ps = z.permute(1,0,2,3).reshape(16,ps.shape[1], ps.shape[2])
		# [RGGB RGGB RGGB RGGB]
		#b12 = F.pixel_shuffle(ps, 2)
		#b12 = F.pixel_shuffle(b12, 2)
		
		if self.all:
			patches = []
			w = 224
			h = 224
			for y in range(0, b12.shape[1]-h, h-32):
				for x in range(0, b12.shape[2]-w, w-32):
					patches.append(b12[:,y:y+h,x:x+w])
			raw = torch.stack(patches)
		else:
			#ps[1::4,:,:] = (ps[1::4,:,:] + ps[2::4,:,:]) / 2
			#ps[2::4,:,:] = ps[3::4,:,:]
			#c = ps.clone()
			#ps[3:6,:,:] = c[4:7,:,:]
			#ps[6:9,:,:] = c[8:11,:,:]
			#ps[9:12,:,:] = c[12:15,:,:]
			#ps = ps[:12]

			#assert ps.shape[0] == 12
			assert ps.shape[0] == 16
			assert ps.shape[1] == b12.shape[1] // 4
			assert ps.shape[2] == b12.shape[2] // 4
		
			img = self.transformGT(ps)
			raw = torch.stack((
				img[ 0, 0::4, 0::4],
				img[ 1, 0::4, 2::4],
				img[ 2, 2::4, 0::4],
				img[ 3, 2::4, 2::4],
				img[ 4, 0::4, 1::4], 
				img[ 5, 0::4, 3::4],
				img[ 6, 2::4, 1::4],
				img[ 7, 2::4, 3::4],
				img[ 8, 1::4, 0::4],
				img[ 9, 1::4, 2::4],
				img[10, 3::4, 0::4],
				img[11, 3::4, 2::4],
				img[12, 1::4, 1::4],
				img[13, 1::4, 3::4],
				img[14, 3::4, 1::4],
				img[15, 3::4, 3::4]))
			

			raw = F.pixel_shuffle(raw,2)
			raw = F.pixel_shuffle(raw,2)

		
		self.inputImage = self.normalize1(raw)
		if self.all: return self.inputImage

		self.outputImage = self.normalize12(img)	
		#print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())
		return self.inputImage, self.outputImage

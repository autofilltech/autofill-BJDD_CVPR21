import glob
import torch.nn as nn
import numpy as np
import time
import cv2
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
import random
class npzDatasetReader(Dataset):
	def __init__(self, image_list, imagePathGT, height, width, transformation=True):
		self.image_list = image_list
		self.imagePathGT = imagePathGT
		self.transformLR = transforms
		self.imageH = height
		self.imageW = width
		self.normalize1 = transforms.Normalize(normMean(1), normStd(1))
		self.normalize12 = transforms.Normalize(normMean(12), normStd(12))

		self.transformHRGT = transforms.Compose([
				])

	
		self.transformRI = transforms.Compose([
				AddGaussianNoise(0, 0.02, pov=1),
				])

	def __len__(self):
		return (len(self.image_list))
	
	def __getitem__(self, i):
		# Read Images
		#print ("print i",i, i+1)
		npz = np.load(self.image_list[i])
		#raw = torch.tensor(npz["raw"]).float() / 255.0
		clean = torch.tensor(npz["gt"]).float() / 255.0
	
		indices = list(range(3))
		random.shuffle(indices)
		cshuf = []
		for i in range(4):
			for idx in indices:
				cshuf.append(clean[i*3+idx,:,:])
				
		img = clean#torch.stack(cshuf)
		
		rr = torch.stack((img[0,0::4,0::4], img[3,0::4,1::4], img[6,1::4,0::4], img[9,1::4,1::4]))
		gr = torch.stack((img[1,0::4,2::4], img[4,0::4,3::4], img[7,1::4,2::4], img[10,1::4,3::4]))
		gb = torch.stack((img[1,2::4,0::4], img[4,2::4,1::4], img[7,3::4,0::4], img[10,3::4,1::4]))
		bb = torch.stack((img[2,2::4,2::4], img[5,2::4,3::4], img[8,3::4,2::4], img[11,3::4,3::4]))
		raw = torch.concat((rr,gr,gb,bb))

		raw = nn.PixelShuffle(4)(torch.stack((
				raw[0], raw[1], raw[4], raw[5],
				raw[2], raw[3], raw[6], raw[7],
				raw[8], raw[9], raw[12], raw[13],
				raw[10], raw[11], raw[14], raw[15])))

		gt = img
		w = 224
		h = 224
		x = random.randrange(0, (gt.shape[2]-w)//4) * 4
		y = random.randrange(0, (gt.shape[1]-h)//4) * 4
		gt = gt[:,y:y+h,x:x+w]
		raw = raw[:,y:y+h,x:x+w]

		#gt = gt[0:3,:,:] + gt[3:6,:,:] + gt[6:9,:,:] + gt[9:12,:,:]
		#gt = gt /

		self.inputImage = self.normalize1(self.transformRI(raw))
		self.gtImageHR = self.normalize12(gt) #self.gtImage #self.transformHRGT(self.gtImage)

		#print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())

		return self.inputImage, self.gtImageHR

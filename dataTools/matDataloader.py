import glob
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
class matDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        self.normalize1 = transforms.Normalize(normMean(1), normStd(1))
        self.normalize3 = transforms.Normalize(normMean(3), normStd(3))

        self.transformHRGT = transforms.Compose([
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomVerticalFlip(0.5),
                                                transforms.ColorJitter((1,1.5), 0.3, (1,2), 0.5),
                                                transforms.RandomRotation(45, Image.BILINEAR),
						transforms.RandomPerspective()
						])

    
        self.transformRI = transforms.Compose([
                                                AddGaussianNoise(0, 0.02, pov=1),
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):
        # Read Images
        #print ("print i",i, i+1)
        raw = loadmat(self.image_list[i])  
        raw = np.asarray(raw["raw"]) 
        raw = np.stack((raw[:,:,0], raw[:,:,1], raw[:,:,3]), 0)
        raw = torch.tensor(raw / 65535) ** (1/2.4)
        raw = self.transformHRGT(raw).float()
        #raw = raw.permute(2, 0, 1)
        # create bayered image straight from the raw original
        # This is for testing a quadbayer format, testing only 1/4 of the channels
        sampled = raw.clone()
        raw = self.normalize3(raw)
        sampled[0, 0::4, 1::4] = 0
        sampled[0, 0::4, 2::4] = 0
        sampled[0, 0::4, 3::4] = 0
        sampled[0, 1::4, :] = 0
        sampled[0, 2::4, :] = 0
        sampled[0, 3::4, :] = 0
	
        sampled[1, 0::4, 0::4] = 0
        sampled[1, 0::4, 1::4] = 0
        sampled[1, 0::4, 3::4] = 0
        sampled[1, 1::4, :] = 0
        sampled[1, 2::4, 1::4] = 0
        sampled[1, 2::4, 2::4] = 0
        sampled[1, 2::4, 3::4] = 0
        sampled[1, 3::4, :] = 0
	
        sampled[2, 2::4, 0::4] = 0
        sampled[2, 2::4, 1::4] = 0
        sampled[2, 2::4, 3::4] = 0
        sampled[2, 0::4, :] = 0
        sampled[2, 1::4, :] = 0
        sampled[2, 3::4, :] = 0
	
	#self.sampledImage = Image.fromarray(np.array((sampled * 255).permute(1,2,0), dtype=np.uint8))

        # test storing it in a single channel instead
        sampled = torch.stack((sampled[0] + sampled[1] + sampled[2],), 0)
	#self.sampledImage = Image.fromarray(np.array((sampled * 255), dtype=np.uint8))
	
        # This is for creating a regular bayer pattern, instead of the code above
        #sampled[0, 0::2, 1::2] = 0
        #sampled[0, 1::2, :] = 0
        #sampled[1, 0::2, 0::2] = 0
        #sampled[1, 1::2, 1::2] = 0
        #sampled[2, 0::2, :] = 0
        #sampled[2, 1::2, 0::2] = 0
        #self.sampledImage = Image.fromarray(np.array((sampled * 255).permute(1,2,0), dtype=np.uint8))

	#self.gtImage = Image.fromarray(np.array((raw * 255).permute(1,2,0), dtype=np.uint8))
        # Transforms Images for training 
        self.inputImage = self.normalize1(self.transformRI(sampled))
        self.gtImageHR = raw #self.gtImage #self.transformHRGT(self.gtImage)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())

        return self.inputImage, self.gtImageHR

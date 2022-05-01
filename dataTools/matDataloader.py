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
        normalize1 = transforms.Normalize(normMean(1), normStd(1))
        normalize3 = transforms.Normalize(normMean(3), normStd(3))

        self.transformHRGT = transforms.Compose([transforms.Resize((self.imageH,self.imageW), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize3,
                                                ])

    
        self.transformRI = transforms.Compose([transforms.ToTensor(),
                                                normalize1,
                                                AddGaussianNoise(0, 0.01, pov=0.6)
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):

        # Read Images
        #print ("print i",i, i+1)
        raw = loadmat(self.image_list[i])  
        raw = np.stack((np.asarray(raw["raw"])[:,:,0], np.asarray(raw["raw"])[:,:,1], np.asarray(raw["raw"])[:,:,3]), 0)
        raw = torch.tensor(raw / 65535.0) ** (1/2.4)
        
        # create bayered image straight from the raw original
        # This is for testing a 4x4 channel interleaved format, testing only 1 channel of RGGB
	sampled = raw.clone()
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
        sampled = sampled[0] + sampled[1] + sampled[2] #torch.stack((sampled[0] + sampled[1] + sampled[2],), 0)
	self.sampledImage = Image.fromarray(np.array((sampled * 255), dtype=np.uint8))
	
	# This is for creating a regular bayer pattern, instead of the code above
	#sampled[0, 0::2, 1::2] = 0
	#sampled[0, 1::2, :] = 0
	#sampled[1, 0::2, 0::2] = 0
	#sampled[1, 1::2, 1::2] = 0
	#sampled[2, 0::2, :] = 0
	#sampled[2, 1::2, 0::2] = 0
	#self.sampledImage = Image.fromarray(np.array((sampled * 255).permute(1,2,0), dtype=np.uint8))

        self.gtImage = Image.fromarray(np.array((raw * 255).permute(1,2,0), dtype=np.uint8))

        # Transforms Images for training 
        self.inputImage = self.transformRI(self.sampledImage)
        self.gtImageHR = self.transformHRGT(self.gtImage)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())

        return self.inputImage, self.gtImageHR

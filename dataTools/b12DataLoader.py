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
from .extensions import AFCapture
import random
class b12DatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        self.normalize1 = transforms.Normalize(normMean(1), normStd(1))
        self.normalize12 = transforms.Normalize(normMean(3), normStd(3))

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
        b12 = AFCapture.readB12(self.image_list[i])
        assert len(b12.shape) == 3

        w = 640
        h = 480
        x = random.randrange(0, (b12.shape[2]-w)//4) * 4
        y = random.randrange(0, (b12.shape[1]-h)//4) * 4

        raw = b12[:,y:y+h, x:x+w] ** 2
        raw[:,0::4,0::4] *= 2
        raw[:,1::4,0::4] *= 2
        raw[:,0::4,1::4] *= 2
        raw[:,1::4,1::4] *= 2
        raw[:,2::4,2::4] *= 3
        raw[:,3::4,2::4] *= 3
        raw[:,2::4,3::4] *= 3
        raw[:,3::4,3::4] *= 3
        raw = torch.sqrt(raw)
        raw = raw.clamp(0., 1.)
        self.inputImage = self.normalize1(raw)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())

        return self.inputImage

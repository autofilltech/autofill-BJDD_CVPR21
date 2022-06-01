import os
import sys
import time
import yaml
from   tqdm import tqdm
from   glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from   torchsummary import summary

from   modelDefinitions.attentionGen import AttentionGenerator
from   datasets.b12 import B12Dataset

import warnings
warning.filterwarnings("ignore", category=UserWarning)

assert torch.cuda.is_available()

#if you change this, also change CUDA_DEVICE_IDX in datasets/operator.h
torch.cuda.set_device(1)

def unnorm(x): return (x+1)/2

with torch.no_grad():
	class PIPInference:
		def __init__(self):
			configPath = "pip.yaml"
			with open(configPath,"r") as stream:
				self.config = yaml.load(stream, Loader=yaml.FullLoader)

			self.name = self.config["name"]
			self.checkpoint = "./checkpoints/pip/pipnet4.200.pth"

			self.model = AttentionGenerator(
				self.config["infer"]["in_channels"],
				self.config["infer"]["out_channels"]).cuda()

			self.datasetInfer = B12Dataset(self.config["infer"]["datapath"],
				(self.config["infer"]["height"], self.config["infer"]["width"]),
				rotate=False, scale=False, infer=True)

		def infer(self):
			loaderInfer = DataLoader(self.datasetInfer, 1, shuffle = False)
			loaderInfer = iter(loaderInfer)

			pbBatch = tqdm(range(len(self.datasetInfer)), position=0, desc="INFER")
			for batch in pbBatch:
				lr, name = loaderInfer.next()
				lr = lr.cuda()
			
				pred = (self.model(lr) * 255).to(torch.uint8);
				hr0 = pred[:,0:3,:,:]
				hr1 = pred[:,3:6,:,:]
				hr2 = pred[:,6:9,:,:]
				hr3 = pred[:,9:12,:,:]

				name = ".".join(name.split(".")[:-1])
				outdir = os.path.join(self.config["infer"]["outpath"], name)
				os.makedirs(outdir, exist_ok=True)

				torchvision.io.write_png(hr0, os.path.join(outdir, "hr0.png"))
				torchvision.io.write_png(hr1, os.path.join(outdir, "hr1.png"))
				torchvision.io.write_png(hr2, os.path.join(outdir, "hr2.png"))
				torchvision.io.write_png(hr3, os.path.join(outdir, "hr3.png"))



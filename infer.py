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
from   modules.polar import polarDefaultNormalize
from   modules.common import defaultNormalize, defaultUnnormalize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
			self.checkpoint = "./checkpoints/pip/pip1net.25.pth"

			self.model = AttentionGenerator(
				self.config["infer"]["in_channels"],
				self.config["infer"]["out_channels"]).cuda()

			checkpoint = torch.load(self.checkpoint)
			self.model.load_state_dict(checkpoint["model"])

			self.datasetInfer = B12Dataset(self.config["infer"]["datapath"],
				(self.config["infer"]["height"], self.config["infer"]["width"]),
				rotate=False, scale=False, infer=True, channels = 1)

		def infer(self):
			loaderInfer = DataLoader(self.datasetInfer, 1, shuffle = False)
			loaderInfer = iter(loaderInfer)

			
			pbBatch = tqdm(range(len(self.datasetInfer)), position=0, desc="INFER")
			for batch in pbBatch:
				lr, names = loaderInfer.next()
				lr = polarDefaultNormalize(lr.cuda())
				name = names[0]
				pred = self.model(lr);
				pred = defaultUnnormalize(pred).squeeze(0).clamp(0,1).cpu()

				name = ".".join(name.split(".")[:-1])
				outdir = self.config["infer"]["outpath"]
				os.makedirs(outdir, exist_ok=True)

				c,h,w = pred.shape
				
				pred = pred.repeat(3,1,1)
				print(pred.shape)
				print (pred.shape)
				torchvision.io.write_png((pred * 255).to(torch.uint8), os.path.join(outdir, name + ".png"))

	PIPInference().infer()

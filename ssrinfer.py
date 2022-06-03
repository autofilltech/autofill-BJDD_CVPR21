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

from   modules.naf import NAFSSRNet
from   datasets.ssr import SSRDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

assert torch.cuda.is_available()

#if you change this, also change CUDA_DEVICE_IDX in datasets/operator.h
torch.cuda.set_device(1)

def unnorm(x): return (x+1)/2

with torch.no_grad():
	class SSRInference:
		def __init__(self):
			configPath = "nafssr.yaml"
			with open(configPath,"r") as stream:
				self.config = yaml.load(stream, Loader=yaml.FullLoader)

			self.name = self.config["name"]
			self.checkpoint = self.config["infer"]["checkpoint"]

			self.model = NAFSSRNet(2, 3).cuda()

			checkpoint = torch.load(self.checkpoint)
			self.model.load_state_dict(checkpoint["model"])

			self.datasetInfer = SSRDataset(self.config["infer"]["datapath"],
				(self.config["infer"]["height"], self.config["infer"]["width"]),
				infer=True)

		def infer(self):
			loaderInfer = DataLoader(self.datasetInfer, 1, shuffle = False)
			loaderInfer = iter(loaderInfer)

			
			pbBatch = tqdm(range(len(self.datasetInfer)), position=0, desc="INFER")
			for batch in pbBatch:
				lr, names = loaderInfer.next()
				lr = lr.cuda()
				name = names[0]
				pred = self.model(lr);
				L,R = pred.squeeze(0).clamp(0,1).cpu().chunk(2)

				name = ".".join(name.split(".")[:-1])
				outdir = os.path.join(self.config["infer"]["outpath"], str(batch))
				os.makedirs(outdir, exist_ok=True)
	
				torchvision.io.write_png((L * 255).to(torch.uint8), os.path.join(outdir, "hhr0.png"))
				torchvision.io.write_png((R * 255).to(torch.uint8), os.path.join(outdir, "hhr1.png"))

	SSRInference().infer()

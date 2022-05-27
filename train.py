import os
import sys
import glob
import time
import yaml
import colorama
from   colorama import Fore, Style
from   tqdm import tqdm
from   glob import glob

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from   torchsummary import summary
from   torch.utils.tensorboard import SummaryWriter
from   torch.profiler import profile, record_function, ProfilerActivity

from modules.common import *
from modules.reduce import *
from modules.naf import NAFNet, NAFSSRNet
from modules.perceptual import FeatureLoss, ColorLoss
from modules.adversarial import AdversarialLoss

from datasets.ssr import SSRDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

assert torch.cuda.is_available()
torch.cuda.set_device(1)


class NAFSSRLoss(nn.Module):
	def __init__(self, channels):
		super(NAFSSRLoss, self).__init__()
		self.lossFunction1 = nn.MSELoss().cuda()
		self.lossFunction2 = FeatureLoss().cuda()
		
	def forward(self, x, y):
		loss = self.lossFunction1(x,y)
		loss += 0.2 * self.lossFunction2(x,y)
		return loss

class PIPLoss(nn.Module):
	def __init__(self, channels):
		super(NAFSSRLoss, self).__init__()
		self.lossFunction1 = nn.MSELoss().cuda()
		self.lossFunction2 = FeatureLoss().cuda()
		self.lossFunction3 = ColorLoss().cuda()
		self.lossFunction4 = AdversarialLoss(channels).cuda()
		
	def forward(self, x, y):
		loss = self.lossFunction1(x,y)
		     + self.lossFunction2(x,y)
		     + self.lossFunction3(x,y)
		     + 0.001 * self.lossFunction4(x,y)
		return loss

class NAFSSRTrainer:
	def __init__(self):
		torch.random.seed()
		
		configPath = "nafssr.yaml"
		with open(configPath,"r") as stream:
			self.config = yaml.load(stream, Loader=yaml.FullLoader)

		self.name = self.config["name"]
		self.checkpointPath = "./checkpoints"
		self.numEpochs = self.config["train"]["numEpochs"]
		self.batchesPerEpoch = self.config["train"]["numBatches"]
		self.startEpoch = 0

		self.model = NAFSSRNet(
				self.config["train"]["scale"],
				self.config["train"]["channels"],
				self.config["train"]["views"],
				self.config["train"]["model"]["width"],
				self.config["train"]["model"]["blocks"]).cuda()
		
		self.optim = optim.Adam(self.model.parameters(), 
				self.config["train"]["optim"]["lr"], 
				self.config["train"]["optim"]["betas"])

		self.sched = optim.lr_scheduler.MultiStepLR(self.optim, 
				self.config["train"]["sched"]["milestones"],
				self.config["train"]["sched"]["gamma"])

		self.datasetTrain = SSRDataset(self.config["train"]["datapath"]["train"], 128, 
				length = self.config["train"]["numBatches"] * self.config["train"]["batchSize"])
		
		self.datasetValidate = SSRDataset(self.config["train"]["datapath"]["validate"], 128)

		self.lossFunction = NAFSSRLoss(self.config["train"]["channels"] * self.config["train"]["views"])

		# warmup
		t = torch.rand((
				self.config["train"]["batchSize"],
				self.config["train"]["channels"] *
				self.config["train"]["views"],
				self.config["train"]["height"],
				self.config["train"]["width"])).cuda()
		self.model(t)

		#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
		#	with record_function("model_forward"):
		#		for _ in range(16): self.model(t)
		#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

	def train(self):
		avgLoss = []

		r = range(self.startEpoch, self.numEpochs)
		pbEpoch = tqdm(r, initial=self.startEpoch, total=self.numEpochs, position=0, desc="EPOCH")
		for epoch in pbEpoch:

			#train
			loaderTrain = DataLoader(self.datasetTrain, 
				self.config["train"]["batchSize"],
				shuffle = True)

			loaderTrain = iter(loaderTrain)

			loaderValidate = DataLoader(self.datasetValidate,
				self.config["train"]["batchSize"],
				shuffle = True)

			loaderValidate = iter(loaderValidate)

			pbBatch = tqdm(range(self.batchesPerEpoch), position=1, desc="BATCH", leave=False)
			for batch in pbBatch:
				lr, hr = loaderTrain.next()
				lr = lr.cuda()
				hr = hr.cuda()

				self.optim.zero_grad()

				pred = self.model(lr)
				loss = self.lossFunction(pred, hr)
				loss.backward()
				self.optim.step()
			
				loss = loss.item()
				avgLoss.append(loss)
				if len(avgLoss) > 100: avgLoss.pop(0)
				avg = sum(avgLoss) / len(avgLoss)

				pbBatch.set_postfix({"Loss": loss, "Avg": avg})
				pbEpoch.refresh()


			
			#validate
			with torch.no_grad():
				loss = 0
				numValidateBatches = 8
				for batch in tqdm(range(numValidateBatches), position=1, desc="VALIDATING", leave=False):
					lr, hr = loaderValidate.next()
					lr = lr.cuda()
					hr = hr.cuda()
				
					pred = self.model(lr)
					loss += self.lossFunction(pred, hr).item() / numValidateBatches
				
				# for testing: save images
				if True:
					pred = torch.clamp(pred,0,1)
					interpolated = F.interpolate(lr.detach(), scale_factor=2, mode="bilinear")
					n,c,h,w = interpolated.shape
					grid1 = torchvision.utils.make_grid(torch.cat(interpolated.cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2,c//2,h,w),2)
					grid2 = torchvision.utils.make_grid(torch.cat(hr.detach().cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2,c//2,h,w),2)
					grid3 = torchvision.utils.make_grid(torch.cat(pred.detach().cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2, c//2,h,w),2)
					torchvision.io.write_png((grid1*255).to(torch.uint8), "grid1.png")
					torchvision.io.write_png((grid2*255).to(torch.uint8), "grid2.png")
					torchvision.io.write_png((grid3*255).to(torch.uint8), "grid3.png")

				pbEpoch.set_postfix({"Loss": loss })
				pbEpoch.refresh()

			#checkpoint
			self.sched.step()
			self.saveCheckpoint(epoch+1)
			self.updateTensorBoard()

	def resume(self):
		path = os.path.join(self.checkpointPath, "{}.*.pth".format(self.name))
		files = glob(path)
		epochs = [int(f.split(".")[-2]) for f in glob(path)]
		if len(epochs) > 0:
			epochs.sort()
			self.loadCheckpoint(epochs[-1])
		self.train()

	def loadCheckpoint(self, epoch):
		path = os.path.join(self.checkpointPath, "{}.{}.pth".format(self.name, epoch))
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint["model"])
		self.optim.load_state_dict(checkpoint["optim"])
		self.sched.load_state_dict(checkpoint["sched"])
		self.lossFunction.load_state_dict(checkpoint["loss"])
		self.startEpoch = checkpoint["epoch"]
		pass

	def saveCheckpoint(self, epoch):
		checkpoint = {
			"epoch": epoch,
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"sched": self.sched.state_dict(),
			"loss" : self.lossFunction.state_dict()}

		path = os.path.join(self.checkpointPath, "{}.{}.pth".format(self.name, epoch))
		torch.save(checkpoint, path)
		pass

	def updateTensorBoard(self):
		pass

NAFSSRTrainer().resume()
exit()


class PIPTrainer:
	def __init__(self):
		torch.random.seed()
		
		configPath = "pip.yaml"
		with open(configPath,"r") as stream:
			self.config = yaml.load(stream, Loader=yaml.FullLoader)

		self.name = self.config["name"]
		self.checkpointPath = "./checkpoints/pip"
		self.numEpochs = self.config["train"]["numEpochs"]
		self.batchesPerEpoch = self.config["train"]["numBatches"]
		self.startEpoch = 0

		self.model = AttentionGenerator(self.config["train"]["in_channels"], self.config["train"]["out_channels"]).cuda()
		
		self.optim = optim.Adam(self.model.parameters(), 
				self.config["train"]["optim"]["lr"], 
				self.config["train"]["optim"]["betas"])

		self.sched = optim.lr_scheduler.MultiStepLR(self.optim, 
				self.config["train"]["sched"]["milestones"],
				self.config["train"]["sched"]["gamma"])

		self.loss = PIPLoss(self.config["train"]["out_channels"])
		
		self.datasetTrain = SSRDataset(self.config["train"]["datapath"]["train"], 128, 
				length = self.config["train"]["numBatches"] * self.config["train"]["batchSize"])
		
		self.datasetValidate = SSRDataset(self.config["train"]["datapath"]["validate"], 128)

		self.lossFunction = PIPLoss(self.config["train"]["channels"] * self.config["train"]["views"])

		# warmup
		t = torch.rand((
				self.config["train"]["batchSize"],
				self.config["train"]["channels"] *
				self.config["train"]["views"],
				self.config["train"]["height"],
				self.config["train"]["width"])).cuda()
		self.model(t)

		#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
		#	with record_function("model_forward"):
		#		for _ in range(16): self.model(t)
		#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

	def train(self):
		avgLoss = []

		r = range(self.startEpoch, self.numEpochs)
		pbEpoch = tqdm(r, initial=self.startEpoch, total=self.numEpochs, position=0, desc="EPOCH")
		for epoch in pbEpoch:

			#train
			loaderTrain = DataLoader(self.datasetTrain, 
				self.config["train"]["batchSize"],
				shuffle = True)

			loaderTrain = iter(loaderTrain)

			loaderValidate = DataLoader(self.datasetValidate,
				self.config["train"]["batchSize"],
				shuffle = True)

			loaderValidate = iter(loaderValidate)

			pbBatch = tqdm(range(self.batchesPerEpoch), position=1, desc="BATCH", leave=False)
			for batch in pbBatch:
				lr, hr = loaderTrain.next()
				lr = lr.cuda()
				hr = hr.cuda()

				self.optim.zero_grad()

				pred = self.model(lr)
				loss = self.lossFunction(pred, hr)
				loss.backward()
				self.optim.step()
			
				loss = loss.item()
				avgLoss.append(loss)
				if len(avgLoss) > 100: avgLoss.pop(0)
				avg = sum(avgLoss) / len(avgLoss)

				pbBatch.set_postfix({"Loss": loss, "Avg": avg})
				pbEpoch.refresh()

			#validate
			with torch.no_grad():
				loss = 0
				numValidateBatches = 8
				for batch in tqdm(range(numValidateBatches), position=1, desc="VALIDATING", leave=False):
					lr, hr = loaderValidate.next()
					lr = lr.cuda()
					hr = hr.cuda()
				
					pred = self.model(lr)
					loss += self.lossFunction(pred, hr).item() / numValidateBatches
				
				# for testing: save images
				if True:
					pred = torch.clamp(pred,0,1)
					interpolated = F.interpolate(lr.detach(), scale_factor=2, mode="bilinear")
					n,c,h,w = interpolated.shape
					grid1 = torchvision.utils.make_grid(torch.cat(interpolated.cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2,c//2,h,w),2)
					grid2 = torchvision.utils.make_grid(torch.cat(hr.detach().cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2,c//2,h,w),2)
					grid3 = torchvision.utils.make_grid(torch.cat(pred.detach().cpu().chunk(2,dim=1)).reshape(2,n,c//2,h,w).permute(1,0,2,3,4).reshape(n*2, c//2,h,w),2)
					torchvision.io.write_png((grid1*255).to(torch.uint8), "grid1.png")
					torchvision.io.write_png((grid2*255).to(torch.uint8), "grid2.png")
					torchvision.io.write_png((grid3*255).to(torch.uint8), "grid3.png")

				pbEpoch.set_postfix({"Loss": loss })
				pbEpoch.refresh()

			#checkpoint
			self.sched.step()
			self.saveCheckpoint(epoch+1)
			self.updateTensorBoard()

	def resume(self):
		path = os.path.join(self.checkpointPath, "{}.*.pth".format(self.name))
		files = glob(path)
		epochs = [int(f.split(".")[-2]) for f in glob(path)]
		if len(epochs) > 0:
			epochs.sort()
			self.loadCheckpoint(epochs[-1])
		self.train()

	def loadCheckpoint(self, epoch):
		path = os.path.join(self.checkpointPath, "{}.{}.pth".format(self.name, epoch))
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint["model"])
		self.optim.load_state_dict(checkpoint["optim"])
		self.sched.load_state_dict(checkpoint["sched"])
		self.lossFunction.load_state_dict(checkpoint["loss"])
		self.startEpoch = checkpoint["epoch"]
		pass

	def saveCheckpoint(self, epoch):
		checkpoint = {
			"epoch": epoch,
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"sched": self.sched.state_dict(),
			"loss" : self.lossFunction.state_dict()}

		path = os.path.join(self.checkpointPath, "{}.{}.pth".format(self.name, epoch))
		torch.save(checkpoint, path)
		pass

	def updateTensorBoard(self):
		pass
		


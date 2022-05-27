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

from datasets.ssr import SSRDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

assert torch.cuda.is_available()

class Trainer:
	def __init__(self):
		torch.random.seed()
		
		configPath = "config.yaml"
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

		# warmup
		t = torch.rand((
				self.config["train"]["batchSize"],
				self.config["train"]["channels"] *
				self.config["train"]["views"],
				self.config["train"]["height"],
				self.config["train"]["width"])).cuda()
		self.model(t)

		with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_forward"):
				for _ in range(16): self.model(t)

		print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

	def train(self):

		lossFunction = nn.L1Loss().cuda()
		mse100 = []

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
				loss = lossFunction(pred, hr)
				loss.backward()
				self.optim.step()
				mse1 = loss.item()
				
				mse100.append(mse1)
				if len(mse100) > 100: mse100.pop(0)
				avg = sum(mse100) / len(mse100)
				pbBatch.set_postfix({"MSE": mse1, "AVG": avg})
				pbEpoch.refresh()

				# for testing: save images
				#grid1 = torchvision.utils.make_grid(torch.cat(lr.detach().cpu().chunk(2,dim=1)))
				#grid2 = torchvision.utils.make_grid(torch.cat(hr.detach().cpu().chunk(2,dim=1)))
				#grid3 = torchvision.utils.make_grid(torch.cat(pred.detach().cpu().chunk(2,dim=1)))
				#torchvision.io.write_png((grid1*255).to(torch.uint8), "grid1.png")
				#torchvision.io.write_png((grid2*255).to(torch.uint8), "grid2.png")
				#torchvision.io.write_png((grid3*255).to(torch.uint8), "grid3.png")

			
			#validate
			with torch.no_grad():
				loss = 0
				numValidateBatches = 8
				for batch in tqdm(range(numValidateBatches), position=1, desc="VALIDATING", leave=False):
					lr, hr = loaderValidate.next()
					lr = lr.cuda()
					hr = hr.cuda()
					
					pred = self.model(lr)
					loss += lossFunction(pred, hr).item() / numValidateBatches

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
		self.startEpoch = checkpoint["epoch"]
		pass

	def saveCheckpoint(self, epoch):
		checkpoint = {
			"epoch": epoch,
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"sched": self.sched.state_dict()}
		path = os.path.join(self.checkpointPath, "{}.{}.pth".format(self.name, epoch))
		torch.save(checkpoint, path)
		pass

	def updateTensorBoard(self):
		pass

Trainer().resume()
exit()

class BJDD:
	def __init__(self):
		# Preapring model(s) for GPU acceleration
		self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.attentionNet = AttentionGenerator(self.inputC, self.outputC).to(self.device)
		self.discriminator = attentiomDiscriminator(self.outputC).to(self.device)

		# Optimizers
		self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
		self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
		
	def customTrainLoader(self, overFitTest = False):
		
		targetImageList1 = imageList(self.gtPath1)
		targetImageList2 = imageList(self.gtPath2)
		print ("Training Samples (Input1):", self.gtPath1, len(targetImageList1))
		print ("Training Samples (Input2):", self.gtPath2, len(targetImageList2))

		if self.dataSamples:
			targetImageList1 = targetImageList1[:self.dataSamples]
			targetImageList2 = targetImageList2[:self.dataSamples]

		datasetReadder1 = b12DatasetReader(   
				image_list=targetImageList1, 
				imagePathGT=self.gtPath1,
				height = self.imageH,
				width = self.imageW)
		datasetReadder2 = b12DatasetReader(
				image_list=targetImageList2,
				imagePathGT=self.gtPath1,
				height = self.imageH,
				width = self.imageW)

		datasetReadder1.all = True
		self.dataset = datasetReadder2

		self.trainLoader1 = torch.utils.data.DataLoader( dataset=datasetReadder1,
			batch_size=1, 
			shuffle=True)
		self.trainLoader2 = torch.utils.data.DataLoader( dataset=datasetReadder2,
			batch_size=self.batchSize, 
			shuffle=True)
		
		return (self.trainLoader1, self.trainLoader2)

	def imageGrid(self, img):
		print(img.min(), img.max())
		img = self.unNorm(img.detach())
		print(img.min(), img.max())
		n,c,h,w = img.shape
		if img.shape[1] == 16 and img.shape[0] <= 8:
			img = img.reshape((n, 4, 4, h, w))
			img = img.permute(0,2,1,3,4)
			img[:,1] = (img[:,1] + img[:,2]) / 2
			img[:,2] = img[:,3]
			img = img[:,:3]
			img = img.permute(0,2,1,3,4)
			assert img.shape == (n, 4, 3, h, w)
			img[:, :, 0, :, :] *= 1 
			img[:, :, 2, :, :] *= 1
			img = img.permute(1,0,2,3,4).reshape(n * 4, 3, h, w)
			return torchvision.utils.make_grid(img.clamp(0,1) ** 0.5, n)
		elif img.shape[0] <= 8:
			#img = img[:,:3,:,:] ** (1/2.4)
			#img = img[:,:,16:208,16:208]
			#img[ :, 0, :, :] *= 2
			#img[ :, 2, :, :] *= 3
			img = img[:,:,:,:] #,img[:,3:6,:,:],img[:,:6:9,:,:],img[:,9:12,:,:]) ** (1/2.4)
			return torchvision.utils.make_grid(img.clamp(0,1), n)
		else:
			img = img.reshape((n, 4, 4, h, w))
			img = img.permute(0,2,1,3,4)
			img[:,1] = (img[:,1] + img[:,2]) / 2
			img[:,2] = img[:,3]
			img = img[:,:3]
			img = img.permute(0,2,1,3,4)
			assert img.shape == (n, 4, 3, h, w)
			img[:, :, 0, :, :] *= 1
			img[:, :, 2, :, :] *= 1
			img = torch.mean(img,dim=1, keepdim=True)
			print(img.shape)
			#assert img.shape == (n, 1, 3, h, w)
			img = img[:,0,:,16:208,16:208] #,img[:,3:6,:,:],img[:,:6:9,:,:],img[:,9:12,:,:]) ** (1/2.4)
			img = torchvision.utils.make_grid(img.clamp(0,1), 19, padding=0)
			save_image(img, 'result.png')
			return img


	def modelTraining(self, resumeTraining=False, overFitTest=False, dataSamples = None):
		
		if dataSamples:
			self.dataSamples = dataSamples 

		# Losses
		featureLoss = regularizedFeatureLoss(self.device).to(self.device)
		reconstructionLoss1 = torch.nn.L1Loss().to(self.device)
		reconstructionLoss2 = ReconstructionLoss().to(self.device) #torch.nn.L1Loss().to(self.device)
		ssimLoss = MSSSIM().to(self.device)
		colorLoss1 = deltaEColorLoss(normalize=True).to(self.device)
		colorLoss2 = ICColorLoss().to(self.device)
		adversarialLoss = nn.BCELoss().to(self.device)
		polarLoss = ICPolarLoss().to(self.device)

		# Resuming Training
		if resumeTraining == True:
			self.modelLoad(True)
		else:
			#self.modelLoad(False)
			pass

		# Starting Training
		customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
		
		# Initiating steps
		stepsPerEpoch = self.dataSamples//self.batchSize
		self.totalSteps =  int(stepsPerEpoch * self.totalEpoch)
		startTime = time.time()
		
		# Initiating progress bar 
		bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))
		currentStep = self.startSteps
		actualLogPath = os.path.join(self.logPath, "default")
		createDir(actualLogPath)
		graphIsStored = False
		epoch = currentStep // stepsPerEpoch
		if resumeTraining:
			customPrint(Fore.CYAN + "RESUMING EPOCH {}".format(epoch+1), textWidth=self.barLen)

		avgLossED = None
		avgLossEG = None
		avgDecay = 0.9
		while epoch <= self.totalEpoch:
			trainingImageLoader1, trainingImageLoader2 = self.customTrainLoader()
			epoch += 1
			summary = SummaryWriter(actualLogPath)
	   
			iterTime = time.time()
			iter1 = iter(trainingImageLoader1)
			iter2 = iter(trainingImageLoader2)
			# Image Generation
			with torch.no_grad():
				rawInput = iter1.next()
				print(rawInput.shape, rawInput.min(), rawInput.max())
				assert len(rawInput.shape) == 5
				if len(rawInput.shape) == 5:
					rawInput = rawInput[0]
					highResFake = []
					for i in range(0, rawInput.shape[0], 1):
						r = rawInput[i:i+1,:,:,:]
						assert len(r.shape) == 4
						highResFake.append(self.attentionNet(r))
					highResFake = torch.cat(highResFake, 0)
					assert len(highResFake.shape) == 4
					summary.add_image("Epoch result", self.imageGrid(highResFake), currentStep + 1)
					summary.flush()
			if (epoch >= self.totalEpoch):
				exit(1)

			while currentStep < epoch * stepsPerEpoch:
				currentStep += 1
				LRImages2, HRImages2 = iter2.next()

				# Images
				#raw1 = LRImages1.to(self.device)
				#gt1  = HRImages1.to(self.device)
				raw2 = LRImages2.to(self.device)
				gt2  = HRImages2.to(self.device)

				# GAN Variables
				onesConst = torch.ones(raw2.shape[0], 1).to(self.device)
				targetReal = (torch.rand(raw2.shape[0],1) * 0.5 + 0.7).to(self.device)
				targetFake = (torch.rand(raw2.shape[0],1) * 0.3).to(self.device)

				'''
				# Image Generation
				with torch.no_grad():
					assert len(rawInput.shape) == 5
					if len(rawInput.shape) == 5:
						assert rawInput.shape[0] == 1
						highResFake = []
						rawInput = rawInput.squeeze(0)
						for i in range(0, rawInput.shape[0], 4):
							r = rawInput[i:i+4,:,:,:]
							assert len(r.shape) == 4
							highResFake.append(self.attentionNet(r))
						highResFake = torch.cat(highResFake, 0)
						print("highResFake shape", highResFake.shape)
						assert len(highResFake.shape) == 4
					else:
						highResFake = self.attentionNet(rawInput)
				'''
				batch = raw2 #torch.cat((raw1,raw2), 0)

				# Optimaztion of Discriminator
				self.optimizerED.zero_grad()
				self.optimizerEG.zero_grad()
				pred = self.attentionNet(batch)
				#pred1 = pred[:raw1.shape[0]]
				#pred2 = pred[raw1.shape[0]:]
				#gt2 = (pred2[:,0:3,:,:] + pred2[:,3:6,:,:] + pred2[:,6:9,:,:] + pred2[:,9:12,:,:]) / 4
				#gt2x4 = torch.cat((gt2,gt2,gt2,gt2),1)
				gt = gt2 #torch.cat((gt1, gt2))

				assert torch.isfinite(torch.sum(pred))

				lossED = adversarialLoss(self.discriminator(gt), targetReal) + \
						adversarialLoss(self.discriminator(pred.detach()), targetFake)
				lossED.backward()
				self.optimizerED.step()

				# Optimization of generator part 1
				Lr1 = reconstructionLoss1(pred, gt)
				Lf1 = featureLoss(pred, gt) # + featureLoss(gt2)
				Lc1 = colorLoss1(pred, gt) #colorLoss1(pred1, gt1)
				generatorContentLoss =  Lr1 + Lf1 + Lc1
				generatorAdversarialLoss = adversarialLoss(self.discriminator(pred), onesConst)
				loss1 = generatorContentLoss + 1e-3 * generatorAdversarialLoss

				# Optimization of generator part 2
				#Lc2 = torch.tensor(0)
				#Lp2 = polarLoss(pred2)
				#Lr2 = reconstructionLoss2(raw2, pred2)
				#loss2 = 0.1 * Lp2 + 0.01 * Lr2
				#loss = loss1 + loss2

				loss1.backward()
				self.optimizerEG.step()
				
				if avgLossED is None: avgLossED = lossED.item()
				else: avgLossED = avgLossED * avgDecay + lossED.item() * (1-avgDecay)

				if avgLossEG is None: avgLossEG = loss1.item()
				else: avgLossEG = avgLossEG * avgDecay + loss1.item() * (1-avgDecay)

				##########################
				###### Model Logger ######
				##########################   

				# Progress Bar
				bar.numerator = currentStep
				print(Fore.CYAN + "EPOCH {}".format(epoch), 
						Fore.YELLOW + "| progress:", bar, 
						Fore.YELLOW + "| Avg Loss: D:{:.4f} G:{:.4f} |".format(
							avgLossED, avgLossEG),
						end='\r')
				
				# Updating training log
				if currentStep % self.interval == 0:
					
					step = currentStep
					#summary.add_scalar("Loss Polar", Lp2.item(), step)
					summary.add_scalar("Loss Feature", Lf1.item(), step)
					#summary.add_scalar("Loss Color 2", Lc2.item(), step)
					summary.add_scalar("Loss Color 1", Lc1.item(), step)
					#summary.add_scalar("Loss Reconstruction 2", Lr2.item(), step)
					summary.add_scalar("Loss Reconstruction 1", Lr1.item(), step)
					#summary.add_scalar("Loss Total 2", loss2.item(), step)
					summary.add_scalar("Loss Total 1", loss1.item(), step)
					'''if not graphIsStored:			
						summary.add_graph(self.attentionNet, rawInput)
						summary.add_graph(self.discriminator, highResFake)
						graphIsStored = True
					'''
					summary.flush()
					#save_image(self.unNorm(rawInput[0]), 'rawinput.png')
					#save_image(self.unNorm(highResReal[0][:3]), '1_groundTruth.png')
					#save_image(self.unNorm(highResFake[0][:3]), '1_modelOutput.png')
					#save_image(self.unNorm(highResReal[0][3:6]), '2_groundTruth.png')
					#save_image(self.unNorm(highResFake[0][3:6]), '2_modelOutput.png')
					#save_image(self.unNorm(highResReal[0][6:9]), '3_groundTruth.png')
					#save_image(self.unNorm(highResFake[0][6:9]), '3_modelOutput.png')
					#save_image(self.unNorm(highResReal[0][9:12]), '4_groundTruth.png')
					#save_image(self.unNorm(highResFake[0][9:12]), '4_modelOutput.png')

					# Saving Weights and state of the model for resume training 
					#self.savingWeights(currentStep)
				
				if (currentStep) % (self.interval ** 2) == 0 : 
					print()	
					stokes = polarLoss.toStokes(pred)
					yuv = polarLoss.stokesToYuv(stokes)
					rgb = polarLoss.yuvToRgb(yuv)
	
					summary.add_image("Input", self.imageGrid(batch), step)
					summary.add_image("Ground Truth", self.imageGrid(gt), step)
					summary.add_image("Generated Images", self.imageGrid(pred), step)
					summary.add_image("Generated Polar", self.imageGrid(rgb), step)
					summary.flush()
					self.savingWeights(currentStep, True)
			print()
			customPrint(Fore.YELLOW + "EPOCH {} COMPLETE".format(epoch), textWidth=self.barLen)
			summary.close()
			#exit()
			'''
			# Image Generation
			with torch.no_grad():
				self.dataset.all = True
				rawInput = iter2.next()
				print(rawInput.shape)
				assert len(rawInput.shape) == 5
				if len(rawInput.shape) == 5:
					rawInput = rawInput[0]
					highResFake = []
					for i in range(0, rawInput.shape[0], 6):
						r = rawInput[i:i+6,:,:,:]
						assert len(r.shape) == 4
						highResFake.append(self.attentionNet(r))
					highResFake = torch.cat(highResFake, 0)
					assert len(highResFake.shape) == 4
					summary.add_image("Epoch result", self.imageGrid(highResFake))
					summary.flush()
				self.dataset.all = False
			'''
		self.savingWeights(currentStep)
		customPrint(Fore.GREEN + "TRAINING COMPLETE", textWidth=self.barLen)
		while True: time.sleep(1)

	def modelInference(self, testImagesPath = None, outputDir = None, resize = None, validation = None, noiseSet = None, steps = None):
		if not validation:
			self.modelLoad(True)
			print("\nInferencing on pretrained weights.")
		else:
			print("Validation about to begin.")
		if not noiseSet:
			noiseSet = self.noiseSet
		if testImagesPath:
			self.testImagesPath = testImagesPath
		if outputDir:
			self.resultDir = outputDir
		

		modelInference = inference(gridSize=self.binningFactor, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation)

		testImageList = modelInference.testingSetProcessor()
		barVal = ProgressBar(len(testImageList) * len(noiseSet), max_width=int(50))
		imageCounter = 0
		with torch.no_grad():
			for noise in noiseSet:
				#print(noise)
				for imgPath in testImageList:
					img = modelInference.inputForInference(imgPath, noiseLevel=noise).to(self.device)
					output = self.attentionNet(img)
					modelInference.saveModelOutput(output, imgPath, noise, steps)
					imageCounter += 1
					if imageCounter % 2 == 0:
						barVal.numerator = imageCounter
						print(Fore.CYAN + "Image Processd |", barVal,Fore.CYAN, end='\r')
		print("\n")

	def modelSummary(self,input_size = None):
		if not input_size:
			input_size = (self.inputC, self.imageH//self.scalingFactor, self.imageW//self.scalingFactor)

	 
		customPrint(Fore.YELLOW + "AttentionNet (Generator)", textWidth=self.barLen)
		summary(self.attentionNet, input_size =input_size)
		print ("*" * self.barLen)
		print()

		customPrint(Fore.YELLOW + "AttentionNet (Discriminator)", textWidth=self.barLen)
		summary(self.discriminator, input_size =input_size)
		print ("*" * self.barLen)
		print()

		'''flops, params = get_model_complexity_info(self.attentionNet, input_size, as_strings=True, print_per_layer_stat=False)
		customPrint('Computational complexity (Enhace-Gen):{}'.format(flops), self.barLen, '-')
		customPrint('Number of parameters (Enhace-Gen):{}'.format(params), self.barLen, '-')

		flops, params = get_model_complexity_info(self.discriminator, input_size, as_strings=True, print_per_layer_stat=False)
		customPrint('Computational complexity (Enhace-Dis):{}'.format(flops), self.barLen, '-')
		customPrint('Number of parameters (Enhace-Dis):{}'.format(params), self.barLen, '-')
		print()'''

		configShower()
		print ("*" * self.barLen)
	
	def savingWeights(self, currentStep, duplicate=None):
		# Saving weights 
		
		if duplicate:
			customPrint(Fore.GREEN + "Saving weights for step {}".format(currentStep), textWidth=self.barLen)
		checkpoint = { 
						'step' : currentStep,
						'stateDictEG': self.attentionNet.state_dict(),
						'stateDictED': self.discriminator.state_dict(),
						'optimizerEG': self.optimizerEG.state_dict(),
						'optimizerED': self.optimizerED.state_dict(),
						}
		saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath, modelName = self.modelName)
		if duplicate:
			saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath + str(currentStep) + "/", modelName = self.modelName, backup=None)



	def modelLoad(self, resume = False):

		if (resume):
			customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)
			previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
		else:
			previousWeight = loadCheckpoints(os.path.join(self.checkpointPath,"base/"), self.modelName, lastWeights = False)
			customPrint(Fore.RED + "Loading base weight with {} pretraining steps".format(previousWeight['step']), textWidth=self.barLen)
		self.attentionNet.load_state_dict(previousWeight['stateDictEG'])
	#self.discriminator.load_state_dict(previousWeight['stateDictED'])
		self.optimizerEG.load_state_dict(previousWeight['optimizerEG']) 
	#self.optimizerED.load_state_dict(previousWeight['optimizerED']) 
	#self.scheduleLR = previousWeight['schedulerLR']
		self.startSteps = int(previousWeight['step']) if resume else 0
		
		customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)



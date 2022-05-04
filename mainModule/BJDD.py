import caffe2
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
#from torchsummary import summary
from ptflops import get_model_complexity_info
from dataTools.customDataloader import *
from dataTools.b12DataLoader import *
from utilities.torchUtils import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *
from utilities.customUtils import *
from loss.pytorch_msssim import *
from loss.colorLoss import *
from loss.percetualLoss import *
from modelDefinitions.attentionDis import *
from modelDefinitions.attentionGen import *
from torchvision.utils import save_image
from loss.icPolarLoss import ICPolarLoss
from loss.icColorLoss import ICColorLoss
from loss.reconstructionLoss import ReconstructionLoss
class BJDD:
	def __init__(self, config):
		
		# Model Configration 
		self.gtPath = config['gtPath']
		self.checkpointPath = config['checkpointPath']
		self.logPath = config['logPath']
		self.testImagesPath = config['testImagePath']
		self.resultDir = config['resultDir']
		self.modelName = config['modelName']
		self.dataSamples = config['dataSamples']
		self.batchSize = int(config['batchSize'])
		self.imageH = int(config['imageH'])
		self.imageW = int(config['imageW'])
		self.inputC = int(config['inputC'])
		self.outputC = int(config['outputC'])
		self.scalingFactor = int(config['scalingFactor'])
		self.binningFactor = int(config['binningFactor'])
		self.totalEpoch = int(config['epoch'])
		self.interval = int(config['interval'])
		self.learningRate = float(config['learningRate'])
		self.adamBeta1 = float(config['adamBeta1'])
		self.adamBeta2 = float(config['adamBeta2'])
		self.barLen = int(config['barLen'])
		
		# Initiating Training Parameters(for step)
		self.currentEpoch = 0
		self.startSteps = 0
		self.totalSteps = 0
		self.adversarialMean = 0
		self.PR = 0.0

		# Normalization
		self.unNorm = UnNormalize()

		# Noise Level for inferencing
		self.noiseSet = [5, 10, 15]
		

		# Preapring model(s) for GPU acceleration
		self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.attentionNet = attentionNet(self.inputC, self.outputC).to(self.device)
		self.discriminator = attentiomDiscriminator(self.outputC).to(self.device)

		# Optimizers
		self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
		self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
		
		# Scheduler for Super Convergance
		self.scheduleLR = None
		
	def customTrainLoader(self, overFitTest = False):
		
		targetImageList = imageList(self.gtPath)
		print ("Trining Samples (Input):", self.gtPath, len(targetImageList))

		if overFitTest == True:
			targetImageList = targetImageList[-1:]
		if self.dataSamples:
			targetImageList = targetImageList[:self.dataSamples]

		datasetReadder = b12DatasetReader(   
												image_list=targetImageList, 
												imagePathGT=self.gtPath,
												height = self.imageH,
												width = self.imageW,
											)

		self.trainLoader = torch.utils.data.DataLoader( dataset=datasetReadder,
														batch_size=self.batchSize, 
														shuffle=True
														)
		
		return self.trainLoader

	def imageGrid(self, img):
		img = self.unNorm(img)
		if img.shape[1] == 12 and img.shape[0] == 1:
			img = img.reshape((4,3,img.shape[2], img.shape[3]))
		return torchvision.utils.make_grid(img ** (1. / 2.4))

	def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples = None):
		
		if dataSamples:
			self.dataSamples = dataSamples 

		# Losses
		featureLoss = regularizedFeatureLoss(self.device).to(self.device)
		reconstructionLoss = ReconstructionLoss() #torch.nn.L1Loss().to(self.device)
		ssimLoss = MSSSIM().to(self.device)
		#colorLoss = deltaEColorLoss(normalize=True).to(self.device)
		colorLoss = ICColorLoss().to(self.device)
		adversarialLoss = nn.BCELoss().to(self.device)
		polarLoss = ICPolarLoss().to(self.device)

		# Overfitting Testing
		if overFitTest == True:
			customPrint(Fore.RED + "Over Fitting Testing with an arbitary image!", self.barLen)
			trainingImageLoader = self.customTrainLoader(overFitTest=True)
			self.interval = 1
			self.totalEpoch = 100000
		else:  
			trainingImageLoader = self.customTrainLoader()


		# Resuming Training
		if resumeTraning == True:
			self.modelLoad(True)
			try:
				pass#self.modelLoad()

			except:
				#print()
				customPrint(Fore.RED + "Would you like to start training from sketch (default: Y): ", textWidth=self.barLen)
				userInput = input() or "Y"
				if not (userInput == "Y" or userInput == "y"):
					exit()
		else:
			self.modelLoad(False)

		# Starting Training
		customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
		
		# Initiating steps
		self.totalSteps =  int(len(trainingImageLoader)*self.totalEpoch)
		startTime = time.time()
		
		# Instantiating Super Convergance 
		#self.scheduleLR = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizerEG, max_lr=self.learningRate, total_steps=self.totalSteps)
		# Initiating progress bar 
		bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))
		currentStep = self.startSteps
		actualLogPath = os.path.join(self.logPath, "default")
		createDir(actualLogPath)
		graphIsStored = False
		while currentStep < self.totalSteps:
			summary = SummaryWriter(actualLogPath)
	   
			iterTime = time.time()
			for LRImages in trainingImageLoader:
				
				##############################
				#### Initiating Variables ####
				##############################
				# Updating Steps
				if currentStep > self.totalSteps:
			#self.savingWeights(currentStep)
					customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)
					exit()
				currentStep += 1

				# Images
				rawInput = LRImages.to(self.device)
		#highResReal = HRGTImages.to(self.device)
			  
				# GAN Variables
		#onesConst = torch.ones(rawInput.shape[0], 1).to(self.device)
		#targetReal = (torch.rand(rawInput.shape[0],1) * 0.5 + 0.7).to(self.device)
		#targetFake = (torch.rand(rawInput.shape[0],1) * 0.3).to(self.device)


				##############################
				####### Training Phase #######
				##############################
	
				# Image Generation
				with torch.no_grad():
					highResFake = self.attentionNet(rawInput)
				
				# Optimaztion of Discriminator
		#self.optimizerED.zero_grad()
		#lossED = adversarialLoss(self.discriminator(highResReal), targetReal) + \
				#		 adversarialLoss(self.discriminator(highResFake.detach()), targetFake)
		#lossED.backward()
		#self.optimizerED.step()

				
				# Optimization of generator 
				#self.optimizerEG.zero_grad()
		#Lr = reconstructionLoss(highResFake, highResReal)
		#Lf = featureLoss(highResFake, highResReal)
		#Lc = colorLoss(highResFake, highResReal)
		#generatorContentLoss =  Lr + Lf + Lc

				# TODO losses
				#Lp = polarLoss(highResFake) * 100
				#Lc = colorLoss(highResFake) * 100
				#Lr = reconstructionLoss(rawInput, highResFake) * 100
		#Lr = reconstructionLoss(rawInput, highResFake)
				#Loss = Lr + Lc + Lp
				#Loss.backward()
		#generatorAdversarialLoss = adversarialLoss(self.discriminator(highResFake), onesConst)
		#lossEG = generatorContentLoss + 1e-3 * generatorAdversarialLoss
		#lossEG.backward()
		#self.optimizerEG.step()

				# Steps for Super Convergance			
				#self.scheduleLR.step()

				##########################
				###### Model Logger ######
				##########################   

				# Progress Bar
				if (currentStep  + 1) % self.interval/2 == 0:
					bar.numerator = currentStep + 1
					print(Fore.YELLOW + "Steps |",bar,Fore.YELLOW + "||",end='\r')
					
				
				# Updating training log
				if (currentStep + 1) % self.interval == 0:
					stokes = polarLoss.toStokes(highResFake)
					print(stokes.shape)
					yuv = polarLoss.stokesToYuv(stokes)
					print(yuv.shape)
					rgb = polarLoss.yuvToRgb(yuv)
					print(rgb.shape)
					step = currentStep + 1
					summary.add_image("Input Images", self.imageGrid(rawInput), step)
					summary.add_image("Generated Images", self.imageGrid(highResFake), step)
					summary.add_image("Generated Polar", self.imageGrid(rgb), step)
			#summary.add_image("GT Images", self.imageGrid(highResReal), step)
					#summary.add_scalar("Loss Polar", Lp.item(), step)
			#summary.add_scalar("Loss Discriminator", lossED.item(), step)
					#summary.add_scalar("Loss Color", Lc.item(), step)
			#summary.add_scalar("Loss Feature", Lf.item(), step)
					#summary.add_scalar("Loss Reconstruction", Lr.item(), step)
					#summary.add_scalar("Loss Total", Loss.item(), step)
					'''if not graphIsStored:			
						summary.add_graph(self.attentionNet, rawInput)
						summary.add_graph(self.discriminator, highResFake)
						graphIsStored = True
					'''
					summary.flush()
			#save_image(self.unNorm(rawInput[0]), 'rawinput.png')
					save_image(self.imageGrid(rgb), "polar.png")
			#save_image(self.unNorm(highResReal[0][:3]), '1_groundTruth.png')
			#save_image(self.unNorm(highResFake[0][:3]), '1_modelOutput.png')
			#save_image(self.unNorm(highResReal[0][3:6]), '2_groundTruth.png')
			#save_image(self.unNorm(highResFake[0][3:6]), '2_modelOutput.png')
			#save_image(self.unNorm(highResReal[0][6:9]), '3_groundTruth.png')
			#save_image(self.unNorm(highResFake[0][6:9]), '3_modelOutput.png')
			#save_image(self.unNorm(highResReal[0][9:12]), '4_groundTruth.png')
			#save_image(self.unNorm(highResFake[0][9:12]), '4_modelOutput.png')


					# TODO save weights
					# Saving Weights and state of the model for resume training 
			#self.savingWeights(currentStep)
				
		#if (currentStep + 1) % (self.interval ** 2) == 0 : 
			#print("\n")
			#self.savingWeights(currentStep + 1, True)
					#self.modelInference(validation=True, steps = currentStep + 1)
			#eHours, eMinutes, eSeconds = timer(iterTime, time.time())
			#print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | LossC: {:.2f}, LossP : {:.2f}, LossEG: {:.2f}, LossED: {:.2f}' 
			#		.format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, colorLoss(highResFake, highResReal), featureLoss(highResFake, highResReal),lossEG, lossED))
					
   
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
		checkpoint = { 
						'step' : currentStep + 1,
						'stateDictEG': self.attentionNet.state_dict(),
						'stateDictED': self.discriminator.state_dict(),
						'optimizerEG': self.optimizerEG.state_dict(),
						'optimizerED': self.optimizerED.state_dict(),
						'schedulerLR': self.scheduleLR
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



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def swish(x):
	return x * torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)



class AdversarialLoss(nn.Module):
	def __init__(self, channels, lr=0.01, betas=[0.9, 0.99], milestones=[], gamma=0.1):
		super(AdversarialLoss, self).__init__()

		self.model = Discriminator(channels)
		self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=betas)
		self.sched = optim.lr_scheduler.MultiStepLR(self.optim, milestones, gamma)
		self.loss = nn.BCELoss()

	def step(self):
		self.sched.step()

	def forward(self, x, y):
		if x.requires_grad:
			targetReal = (torch.rand(x.shape[0],1) * 0.3 + 0.7).to(x.device)
			targetFake = (torch.rand(x.shape[0],1) * 0.3 + 0.0 ).to(x.device)

			self.optim.zero_grad()
			real = self.model(y)
			fake = self.model(x.detach())

			loss = self.loss(real, targetReal) + self.loss(fake, targetFake)
			loss.backward()
			self.optim.step()
	
		ones = torch.ones(x.shape[0], 1).to(x.device)
		fake = self.model(x)
		loss = self.loss(fake, ones)
		return loss



import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torchsummary import summary
class BayerUpsample4x4(nn.Module):
	def __init__(self):
		super(BayerUpsample4x4, self).__init__()
		self.conv = nn.Conv2d(16,16,7,padding=3,groups=16, bias=False)
		kernel = torch.tensor([[0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]])
		kernel = torch.mul(kernel, kernel.T)
		assert kernel.shape == (7,7)
		print(self.conv.weight.shape)
		self.conv.weight = nn.Parameter(kernel.expand(16, 1, 7, 7))
		self.conv.weight.requires_grad = False
	
	def forward(self,x):
		x = F.pixel_unshuffle(x, 2)
		x = F.pixel_unshuffle(x, 2)
		n,c,h,w = x.shape
		y = torch.zeros((n, c, h*4, w*4)).cuda()
		y[:, 0, 0::4, 0::4] = x[:,0]
		y[:, 1, 0::4, 2::4] = x[:,1]
		y[:, 2, 2::4, 0::4] = x[:,2]
		y[:, 3, 2::4, 2::4] = x[:,3]
		y[:, 4, 0::4, 1::4] = x[:,4]
		y[:, 5, 0::4, 3::4] = x[:,5]
		y[:, 6, 2::4, 1::4] = x[:,6]
		y[:, 7, 2::4, 3::4] = x[:,7]
		y[:, 8, 1::4, 0::4] = x[:,8]
		y[:, 9, 1::4, 2::4] = x[:,9]
		y[:,10, 3::4, 0::4] = x[:,10]
		y[:,11, 3::4, 2::4] = x[:,11]
		y[:,12, 1::4, 1::4] = x[:,12]
		y[:,13, 1::4, 3::4] = x[:,13]
		y[:,14, 3::4, 1::4] = x[:,14]
		y[:,15, 3::4, 3::4] = x[:,15]

		y = self.conv(y);
		return y

class LocalNormBlock(nn.Module):
	def __init__(self, kernel_size = 3, dilation = 1):
		super(LocalNormBlock, self).__init__()
		self.conv = nn.Conv2d(1, 1, kernel_size, dilation = dilation, padding = dilation, bias = False)
		self.conv.weight = nn.Parameter(torch.ones((1,1,kernel_size,kernel_size)) / (kernel_size ** 2))
		self.conv.weight.requires_grad = False

	def forward(self, x): # [*, 1, H, W] => [*, 2, H, W]
		mean = self.conv(x)
		x = (x - mean)
		x = torch.concat((x, mean), dim=-3)
		return x

class LocalDenormBlock(nn.Module):
	def __init__(self):
		super(LocalDenormBlock,self).__init__()

	def forward(self, x): # [*, C+Cm, H, W] => [*, C, H, W]
		n,c,h,w = x.shape
		x, mean = torch.split(x, (c//2,c//2), dim=-3)
		x = x + mean
		return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def swish(x):
    return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def tensorImgeSplit(targetTensor):
    return targetTensor[:,:1,:,:], targetTensor[:,1:2,:,:], targetTensor[:,2:,:,:]


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class attentionGuidedResBlock(nn.Module):
    def __init__(self, squeezeFilters = 32, expandFilters = 64, dilationRate=1, bn=False, act=False, bias=True):
        super(attentionGuidedResBlock, self).__init__()
        #print ("{} squeeze and {} extraction".format(squeezeFilters, expandFilters))
        self.bn = bn
        self.act = act
        self.depthAtten = SELayer(squeezeFilters)
        self.exConv = nn.Conv2d(squeezeFilters, expandFilters,  1, dilation=dilationRate, padding=0)
        self.exConvBn = nn.BatchNorm2d(expandFilters)
        self.sepConv =  SeparableConv2d(expandFilters, expandFilters,  kernel_size=3, stride=1, dilation=dilationRate, padding=dilationRate)
        self.sepConvBn =  nn.BatchNorm2d(expandFilters)
        self.sqConv = nn.Conv2d(expandFilters, squeezeFilters, 1, dilation=dilationRate, padding=0)
        self.sqConvBn = nn.BatchNorm2d(squeezeFilters)
        
    def forward(self, inputTensor):
        xDA = self.depthAtten(inputTensor)
        if self.bn == True:
            xEx = F.leaky_relu(self.exConvBn(self.exConv(inputTensor)))
            xSp = F.leaky_relu(self.sepConvBn(self.sepConv(xEx)))
            xSq = self.sqConvBn(self.sqConv(xSp))  
        else:
            xEx = F.leaky_relu(self.exConv(inputTensor))
            xSp = F.leaky_relu(self.sepConv(xEx))
            xSq = self.sqConv(xSp)
        return inputTensor + xSq + xDA


class pixelShuffleUpsampling(nn.Module):
    def __init__(self, inputFilters, scailingFactor=2):
        super(pixelShuffleUpsampling, self).__init__()
    
        self.upSample = nn.Sequential(  nn.Conv2d(inputFilters, inputFilters * (scailingFactor**2), 3, 1, 1),
                                        nn.BatchNorm2d(inputFilters * (scailingFactor**2)),
                                        nn.PixelShuffle(upscale_factor=scailingFactor),
                                        nn.PReLU()
                                    )
    def forward(self, tensor):
        return self.upSample(tensor)



        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        #assert not bn
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialAttentionBlock(nn.Module):
    def __init__(self, spatial_filter=32):
        super(SpatialAttentionBlock, self).__init__()
        self.spatialAttenton = SpatialAttention()
        self.conv = nn.Conv2d(spatial_filter, spatial_filter,  3, padding=1)


    def forward(self, x):
        x1 = self.spatialAttenton(x)
        #print(" spatial attention",x1.shape)
        xC = self.conv(x)
        #print("conv",xC.shape)
        y = x1 * xC
        #print("output",y.shape)
        return y

        

#net = SpatialAttentionBlock(64)
#summary(net, input_size = (64,128, 128))
#print ("reconstruction network")
#net = depthAttentiveResBlock(32, 32,1)
#summary(net, input_size = (32, 128, 128))
#print ("reconstruction network")

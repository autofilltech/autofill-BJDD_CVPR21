#normMean = [0.485, 0.456, 0.406]
#normStd = [0.229, 0.224, 0.225]

import numpy as np

def normMean(c):
    return np.full(c, 0.5)

def normStd(c):
    return np.full(c, 0.5)

class UnNormalize(object):

    def __call__(self, tensor, imageNetNormalize=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        mean = normMean(tensor.shape[0])
        std = normStd(tensor.shape[0])
        if imageNetNormalize:
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        else:
            tensor = (tensor * 0.5) + 0.5
        
        return tensor

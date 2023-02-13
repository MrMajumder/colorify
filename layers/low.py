import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convolution import Conv2d

class LowLevelFeatures(nn.Module):
    """Low-Level Features Network"""

    def __init__(self, net_divisor=1):
        super(LowLevelFeatures, self).__init__()

        ksize = np.array([1, 64, 128, 128, 256, 256, 512]) // net_divisor
        ksize[0] = 1
        torch.nn

        self.conv1 = Conv2d(1, ksize[1], 2)
        self.conv2 = Conv2d(ksize[1], ksize[2], 1)
        self.conv3 = Conv2d(ksize[2], ksize[3], 2)
        self.conv4 = Conv2d(ksize[3], ksize[4], 1)
        self.conv5 = Conv2d(ksize[4], ksize[5], 2)
        self.conv6 = Conv2d(ksize[5], ksize[6], 1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        return out

if __name__ == "__main__":
    print("Low Level Network")
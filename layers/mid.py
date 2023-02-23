import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.convolution import Conv2d

class MidLevelFeatures(nn.Module):
    """Mid-Level Features Network"""

    def __init__(self, net_divisor=1):
        super(MidLevelFeatures, self).__init__()

        ksize = np.array([512, 512, 256]) // net_divisor

        self.conv7 = Conv2d(ksize[0], ksize[1], 1)
        self.conv8 = Conv2d(ksize[1], ksize[2], 1)

    def forward(self, x):
        out = F.relu(self.conv7(x))
        out = F.relu(self.conv8(out))
        return out

if __name__ == "__main__":
    print("Mid level Network")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.convolution import Conv2d

class ColorizationNetwork(nn.Module):
    """Colorizaion Network"""

    def __init__(self, net_divisor=1):
        super(ColorizationNetwork, self).__init__()

        ksize = np.array([256, 128, 64, 64, 32]) // net_divisor

        self.conv9 = Conv2d(ksize[0], ksize[1], 1)
        
        # Here comes upsample #1
        
        self.conv10 = Conv2d(ksize[1], ksize[2], 1)
        self.conv11 = Conv2d(ksize[2], ksize[3], 1)
        
        # Here comes upsample #2        
        
        self.conv12 = Conv2d(ksize[3], ksize[4], 1)
        self.conv13 = Conv2d(ksize[4], 2, 1)
    
    def forward(self, x):
        out = F.relu(self.conv9(x))

        # Upsample #1        
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv10(out))
        out = F.relu(self.conv11(out))
        
        # Upsample #2
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        
        # Upsample #3
        out = nn.functional.interpolate(input=out, scale_factor=2)
        
        return out

if __name__ == "__main__":
    print("Colorization network")
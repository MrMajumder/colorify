import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.convolution import Conv2d
from layers.low import LowLevelFeatures
from layers.mid import MidLevelFeatures
from layers.globalF import GlobalFeatures
from layers.classification import ClassificationNetwork
from layers.colorization import ColorizationNetwork

class ColorifyNet(nn.Module):
    """Colorization network class"""

    def __init__(self, num_classes, net_divisor=1):
        """Initializes the network.

        Args:
            net_divisor - divisor of net output sizes. Useful for debugging.
        """
        super(ColorifyNet, self).__init__()

        self.net_divisor = net_divisor

        self.conv_fuse = Conv2d(512 // net_divisor, 256 // net_divisor, 1, kernelSize=1, padding=0)

        self.low = LowLevelFeatures(net_divisor)
        self.mid = MidLevelFeatures(net_divisor)
        self.classifier = ClassificationNetwork(num_classes, net_divisor)
        self.glob = GlobalFeatures(net_divisor)
        self.col = ColorizationNetwork(net_divisor)



    def fusion_layer(self, mid_out, glob_out):
        h = mid_out.shape[2]  # Height of a picture  
        w = mid_out.shape[3]  # Width of a picture
        
        glob_stack2d = torch.stack(tuple(glob_out for _ in range(w)), 1)
        glob_stack3d = torch.stack(tuple(glob_stack2d for _ in range(h)), 1)
        glob_stack3d = glob_stack3d.permute(0, 3, 1, 2)

        # 'Merge' two volumes
        stack_volume = torch.cat((mid_out, glob_stack3d), 1)

        out = F.relu(self.conv_fuse(stack_volume))
        return out


    def forward(self, x):
        # Low level
        low_out = self.low(x)
        
        # Net branch         
        mid_out = low_out
        glob_out = low_out

        # Mid level
        mid_out = self.mid(mid_out)

        # Global
        glob_out, classification_in = self.glob(glob_out)

        # Classification
        classification_out = self.classifier(classification_in)

        # Fusion layer
        out = self.fusion_layer(mid_out, glob_out)

        # Colorization Net
        out = self.col(out)
        
        return out, classification_out
        

if __name__ == "__main__":
    print("Main network architecture")
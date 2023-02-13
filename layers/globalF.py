import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convolution import Conv2d

class GlobalFeatures(nn.Module):
    """Global Features Network"""

    def __init__(self, net_divisor=1):
        super(GlobalFeatures, self).__init__()

        ksize = np.array([512, 1024, 512, 256]) // net_divisor
        self.ksize0 = ksize[0]

        self.conv1 = Conv2d(ksize[0], ksize[0], 2)
        self.conv2 = Conv2d(ksize[0], ksize[0], 1)
        self.conv3 = Conv2d(ksize[0], ksize[0], 2)
        self.conv4 = Conv2d(ksize[0], ksize[0], 1)
        self.fc1 = nn.Linear(7*7*ksize[0], ksize[1])
        self.fc2 = nn.Linear(ksize[1], ksize[2])
        self.fc3 = nn.Linear(ksize[2], ksize[3])

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = y.view(-1, 7*7*self.ksize0)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        
        # Branching
        out = y
        classification_in = y

        out = F.relu(self.fc3(out))

        return out, classification_in

if __name__ == "__main__":
    print("Global Features Network")
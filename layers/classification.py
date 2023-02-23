import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassificationNetwork(nn.Module):
    """Classification Network Class"""

    def __init__(self, num_classes, net_divisor=1):
        super(ClassificationNetwork, self).__init__()
        
        self.num_classes = num_classes
        ksize = np.array([512, 256]) // net_divisor

        self.fc1 = nn.Linear(ksize[0], ksize[1])
        self.fc2 = nn.Linear(ksize[1], num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    print("Classification Network")
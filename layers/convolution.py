import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Conv2d(inChannel, outChannel, stride, kernelSize=3, padding=1):
    """Returns an instance of nn.Conv2d"""
    return nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                         stride=stride, kernel_size=kernelSize, padding=padding)

if __name__ == "__main__":
    print("Convolution Layer")
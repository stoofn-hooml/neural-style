from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StylizationNetwork(nn.Module):
    """docstring."""

    def conv_block(in_channels, out_channels, kernel_size, stride):
        return nn.sequential(
            nn.conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels), # ???
            nn.ReLU()
        )

    def res_block(in_channels, out_channels, kernel_size, stride):
        return nn.sequential(
            conv_block(in_channels, out_channels, kernel_size, stride)
            conv_block(in_channels, out_channels, kernel_size, stride)
        )


    def __init__(self):
        """docstring."""
        super(StylizationNetwork, self).__init__()

        #  architecture as specified in Huang paper

        #  (in_channels, out_channels, kernel_size (filter), stride)
        self.conv_block_1 = conv_block(3, 16, 3, 1)
        self.conv_block_2 = conv_block(16, 32, 3, 2)
        self.conv_block_3 = conv_block(32, 48, 3, 2)

        self.res_block_1 = res_block(48, 48, 3, 1)

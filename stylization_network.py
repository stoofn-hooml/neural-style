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

    def conv_block(in_channels, out_channels, kernel_size, stride, activation, transpose=False):
        if activation == 'ReLu':
            act = nn.Relu()
        elif activation == 'Tanh':
            act = nn.Tanh()

        if activation == '':
            return nn.sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels)
            )
        elif transpose: #for deconvolution blocks
            return nn.sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels), # ???
                act
            )
        else: #for cnormal onvolution blocks
            return nn.sequential(
                nn.conv2d(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels), # ???
                act
            )

    def res_block(in_channels, out_channels, kernel_size, stride):
        return nn.sequential(
            conv_block(in_channels, out_channels, kernel_size, stride, 'ReLu'),
            conv_block(in_channels, out_channels, kernel_size, stride, '')
        )


    def __init__(self):
        """docstring."""
        super(StylizationNetwork, self).__init__()

        #  architecture as specified in Huang paper section 3.1

        #  (in_channels, out_channels, kernel_size (filter), stride)
        self.conv_block_1 = conv_block(3, 16, 3, 1, 'Relu')
        self.conv_block_2 = conv_block(16, 32, 3, 2, 'Relu')
        self.conv_block_3 = conv_block(32, 48, 3, 2, 'Relu')

        self.res_block_1 = res_block(48, 48, 3, 1)
        self.res_block_2 = res_block(48, 48, 3, 1)
        self.res_block_3 = res_block(48, 48, 3, 1)
        self.res_block_4 = res_block(48, 48, 3, 1)
        self.res_block_5 = res_block(48, 48, 3, 1)

        # 'deconvolutional blocks' are equivalent to transposed conv blocks
        # 0.5 stride in deconv translates to 2 stride in conv
        self.deconv_block_1 = conv_block(48, 32, 3, 2, 'Relu', True)
        self.deconv_block_2 = conv_block(32, 16, 3, 2, 'Relu', True)

        self.conv_block_4 = conv_block(16, 3, 3, 1, 'Tanh')

    def forward(self, content):
        conv1 = self.conv_block_1(content)
        conv2 = self.conv_block_2(conv1)
        conv3 = self.conv_block_3(conv2)
        res1 = self.res_block_1(conv3)
        res2 = self.res_block_1(res1)
        res3 = self.res_block_1(res2)
        res4 = self.res_block_1(res3)
        res5 = self.res_block_1(res4)
        deconv1 = self.deconv_block_1(res5)
        deconv2 = self.deconv_block_2(deconv1)
        conv4 = self.conv_block_4(deconv2)
        return conv4

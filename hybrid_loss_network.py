from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


""" code skeleton taken from Pytorch tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentLoss(nn.Module):
    """ computes loss between content (target) image and generated (input) using MSE for spatial loss"""

    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input, generated):
        b, c, h, w = input.shape

        self.loss = (1 /(c * h * w)) * F.mse_loss(input, generated)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """ computes loss between style image and generated image using Gram matrix for spatial loss """

    def __init__(self, style):
        super(StyleLoss, self).__init__()
        self.style = style

    def forward(self, generated):
        generated_G = gram_matrix(generated)
        style_G = gram_matrix(self.style)

        numChannels = generated.shape[3]
        self.loss = (1 / numChannels ** 2) * F.mse_loss(generated_G, style_G)
        return generated

class TVLoss(nn.Module):
    """ TV regularization factor for spatial loss (Huang section 3.2.1) """
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input): #calculates overall pixel smoothness for handling checkerboard artifacts
        b, c, h, w = input.shape

        sum = 0
        for i_c in range(c):
            for i_h in range(h-1):
                for i_w in range(w-1):
                    sum += (input[0][i_c][i_h][i_w+1] - input[0][i_c][i_h][i_w]) ** 2
                    sum += (input[0][i_c][i_h+1][i_w] - input[0][i_c][i_h][i_w]) ** 2

        return sum ** 0.5


class TemporalLoss(nn.Module): # TODO: Rename variables
    """
    computes loss between consecutive generated frame
    x: frame t
    flow_t1: optical flow(frame t-1)
    mask: confidence mask of optical flow
    """
    def __init__(self, gpu):
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, flow_t1, mask):
        assert x.shape == flow_t1, "inputs are not the same"
        x = x.view(1, -1)
        flow_t1 = flow_t1.view(1, -1)
        mask = mask.view(-1)

        D = flow_t1.shape[1]
        return (1 / D) * mask * x, flow_t1


class Normalization(nn.Module):
    """ module to normalize input image so we can easily put it in a nn.Sequential """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


""" Defaults for hybrid loss network """

# desired depth layers to compute style/content losses :
content_layers_default = ['relu_10']
style_layers_default = ['relu_2', 'relu_4', 'relu_6', 'relu_10']

vgg = models.vgg19(pretrained=True).features.to(device).eval()

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def get_spatial_loss_network(style_img, vgg=vgg,
                    normalization_mean=vgg_normalization_mean,
                    normalization_std=vgg_normalization_std,
                    content_layers=content_layers_default,
                    style_layers=style_layers_default):
    vgg = copy.deepcopy(vgg)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []


    # assuming that vgg is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    spatial_loss_network = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        spatial_loss_network.add_module(name, layer)
        print(name)

        if name in content_layers:
            # add content loss:
            # target = spatial_loss_network(content_img).detach()
            content_loss = ContentLoss()
            spatial_loss_network.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            style = spatial_loss_network(style_img).detach()
            style_loss = StyleLoss(style)
            spatial_loss_network.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    # for i in range(len(spatial_loss_network) - 1, -1, -1):
    #     if isinstance(spatial_loss_network[i], ContentLoss) or isinstance(spatial_loss_network[i], StyleLoss):
    #         break

    # spatial_loss_network = spatial_loss_network[:(i + 1)]

    return spatial_loss_network, style_losses, content_losses

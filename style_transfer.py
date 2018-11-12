"""

ARCHITECTURE:

1: Stylizing Network
custom cnn (nn.sequential)
input - frame 1, frame 2
backprop - gradients from hybrid loss
output - stylized frame 1, stylized frame 2

2: Temporal Loss
a nn.module?
input - stylized frame 1, stylized frame 2
Deepflow function - mse between frame 1 (warped by optical flow) and frame 2
output - a value

3: Spatial Loss 2x
VGG 19 CNN (nn.sequential)
input 1 - stylized frame 1, content frame 1, style image
output 1 - a value
input 2 -  stylized frame 2, content frame 2, style image
output 2 - a value

4: hybrid loss
a nn.module?
input - spatial loss, temporal loss
output - a value
gradients used for stylizing network

"""

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

# our imports
from stylization_network import StylizationNetwork
from hybrid_loss_network import get_spatial_loss_network, TemporalLoss, TVLoss
from dataset import get_loader
from opticalflow import opticalflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./images/picasso.jpg")
# content_img = image_loader("./images/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are update


plt.figure()
imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')


# add the original input image to the figure:
# plt.figure()
# imshow(generated, title='Input Image')

def get_input_optimizer(stylization_network):
    # this line to show that input is a parameter that requires a gradient

    # Adam???
    optimizer = optim.LBFGS(stylization_network.parameters())
    return optimizer

# hyperperameter defaults (specified in section 4.1)
default_content_weight = 1
default_style_weight = 10
default_temporal_weight = 10000
default_variation_weight = .001

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       style_img, num_steps=300,
                       style_weight=default_style_weight,
                       content_weight=default_content_weight,
                       temporal_weight=default_temporal_weight
                       variation_weight=default_variation_weight):
    """Run the style transfer."""
    print('Building the style transfer model..')

    # initialize starting values/networks

    # generated = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # generated = torch.randn(content_img.data.size(), device=device)

    stylization_network = StylizationNetwork()

    spatial_loss_network, style_losses, content_losses = get_spatial_loss_network(style_img)

    tv = TVLoss() #TODO: initialize TV with generated frame

    temporal_loss = TemporalLoss()

    optimizer = get_input_optimizer(stylization_network) # TODO: Replace with ADAM (see section 4.1)

    print('Optimizing..')
    run = [0]

    # get loader of video

    while run[0] <= num_steps:
        # loader is an iterator so it must be accessed with enumerate()
        # each element of enumerate(loader) is a list of frames (a video)
        for unused, frames in enumerate(loader):
            # loop through all the frames for each video
            for i in range(1, len(frames)):
                def closure():
                    content_t = frames[i];      # current frame
                    content_t1 = frames[i-1];   # previous frame

                    generated_t = stylization_network(content_t)
                    generated_t1 = stylization_network(content_t1)

                    # correct the values of updated input image
                    # generated.data.clamp_(0, 1)

                    # clears gradients for each iteration of backprop
                    optimizer.zero_grad()


                    # calculate style/content loss for current frame (content_t)
                    spatial_loss_network(content_t, generated_t) #generated is modified in place
                    style_loss_t = 0
                    content_loss_t = 0

                    for sl in style_losses:
                        style_loss_t += sl.loss
                    for cl in content_losses:
                        content_loss_t += cl.loss


                    # calculate style/content loss for previous frame (content_t1)
                    spatial_loss_network(content_t1, generated_t1)
                    style_loss_t1 = 0
                    content_loss_t1 = 0

                    for sl in style_losses:
                        style_loss_t1 += sl.loss
                    for cl in content_losses:
                        content_loss_t1 += cl.loss

                    # total style loss (section 3.2.1)
                    total_style_loss = style_loss_t + style_loss_t1
                    total_style_loss *= style_weight

                    # total content loss (section 3.2.1)
                    total_content_loss = content_loss_t + content_loss_t1
                    total_content_loss *= content_weight

                    # regularization (TV Regularizer, section 3.2.1)
                    tv_loss = tv(generated_t)
                    tv_loss *= variation_weight

                    # final spatial loss
                    spatial_loss = total_style_loss + total_content_loss + tv_loss

                    # Optical flow (Temporal Loss, section 3.2.2)
                    flow_t1, mask = opticalflow(generated_t.data.numpy(), generated_t1.data.numpy())

                    temporal_loss = temporal_loss(generated_t, flow_t1, mask)
                    temporal_loss *= temporal_weight

                    # Hybrid loss and backprop
                    hybrid_loss = spatial_loss + temporal_loss
                    hybrid_loss.backward()

                    run[0] += 1
                    if run[0] % 50 == 0:
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f} Temporal Loss: {:4f}'.format(
                            total_style_loss.item(), total_content_loss.item(), temporal_loss.item()))
                        print()

                    # return style_loss + content_loss #old final loss
                    return spatial_loss + temporal_loss # backpropogated hybrid loss

                optimizer.step(closure)

        # a last correction...
        generated.data.clamp_(0, 1)

        return generated


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, generated, 300, 1000000, 1)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

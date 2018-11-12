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
from hybrid_loss_network import get_loss_network
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


# generated = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# generated = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
# plt.figure()
# imshow(generated, title='Input Image')

def get_input_optimizer(stylization_network):
    # this line to show that input is a parameter that requires a gradient

    # Adam???
    optimizer = optim.LBFGS(stylization_network.parameters())
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       style_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')

    stylization_network = StylizationNetwork()

    get_spatial_loss_network, style_losses, content_losses = get_spatial_loss_network(style_img, cnn,
        normalization_mean, normalization_std)

    optimizer = get_input_optimizer(stylization_network)

    print('Optimizing..')
    run = [0]

    # get loader of video

    while run[0] <= num_steps:
        # loader is an iterator so it must be accessed with enumerate()
        # each element of enumerate(loader) is a list of frames (a video)
        for _, frames in enumerate(loader):
            # loop through all the frames for each video
            for i in range(1, len(frames)):
                def closure():
                    content_t = frames[i];      # current frame
                    content_t1 = frames[i-1];   # previous frame

                    generated_t = stylizationNetwork(content_t)
                    generated_t1 = stylizationNetwork(content_t1)

                    # correct the values of updated input image
                    # generated.data.clamp_(0, 1)

                    # clears gradients for each iteration of backprop
                    optimizer.zero_grad()


                    # calculate losses for content_t
                    spatial_loss_network(generated_t, content_t) #generated is modified in place
                    style_score_t = 0
                    content_score_t = 0

                    for sl in style_losses:
                        style_score_t += sl.loss
                    for cl in content_losses:
                        content_score_t += cl.loss


                    # calculate losses for content_t1
                    spatial_loss_network(generated_t1, content_t1)
                    style_score_t1 = 0
                    content_score_t1 = 0

                    for sl in style_losses:
                        style_score_t1 += sl.loss
                    for cl in content_losses:
                        content_score_t1 += cl.loss

                    total_style_score = style_score_t + style_score_t1
                    total_style_score *= style_weight

                    total_content_score = content_score_t + content_score_t1
                    total_content_score *= content_weight

                    # Optical flow
                    flow, mask = opticalflow(generated_t.data.numpy(), generated_t1.data.numpy())

                    temporal_score = TemporalLoss(generated_t, flow, mask)
                    temporal_score *= temporal_weight

                    loss = total_style_score + total_content_score + temporal_score
                    loss.backward()

                    run[0] += 1
                    if run[0] % 50 == 0:
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                            style_score.item(), content_score.item()))
                        print()

                    return style_score + content_score

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

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
from hybrid_loss_network import SpatialLossNetwork, ContentLoss, StyleLoss, TVLoss, TemporalLoss
from dataset import get_loader
from opticalflow import opticalflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use shape instead of size so we can be sure that content frames will be the same
img_shape = (640, 360) if torch.cuda.is_available() else (128, 72)

transform = transforms.Compose([
    transforms.Resize(img_shape),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

# resize, convert to tensor, add dimension, put on GPU if available
def transform_img(image, style):
    # fake batch dimension required to fit network's input dimensions
    image = transform(image) # adds another dimension to tensor
    if (style):
        image = image.unsqueeze(0) # adds another dimension to style tensor
    return image.to(device, torch.float)


style_img = transform_img(Image.open("./images/picasso.jpg"), True)
content_path = './content_videos'
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

model_path = "./models/model.pth"

# hyperperameter defaults (specified in section 4.1)
default_content_weight = 1
default_style_weight = 10
default_temporal_weight = 10000
default_variation_weight = .001

# style_loss_layers = [1]

def train_stylization_network(style_img, num_steps=200,
                       content_weight=default_content_weight,
                       style_weight=default_style_weight,
                       temporal_weight=default_temporal_weight,
                       variation_weight=default_variation_weight):
    """Run the style transfer."""
    print('Building the style transfer model...')

    # initialize starting values/networks
    stylization_network = StylizationNetwork()
    spatial_loss_network = SpatialLossNetwork()

    content_loss = ContentLoss(device=='cuda')
    style_loss = StyleLoss(device=='cuda')
    tv_loss = TVLoss()
    temporal_loss = TemporalLoss()

    optimizer = optim.Adam(stylization_network.parameters())

    print('Optimizing...')
    steps_completed = 0

    # get loader of video
    video_loader = get_loader(1, content_path, transform_img)

    while steps_completed <= num_steps:
        # video_loader is an iterator so it must be accessed with enumerate()
        # each element of enumerate(video_loader) is a list of frames (a video)
        for _, frames in enumerate(video_loader):
            # loop through all the frames for each video
            for i in range(1, len(frames)):
                print(i)

                content_t = frames[i];      # current frame
                content_t1 = frames[i-1];   # previous frame

                generated_t = stylization_network(content_t)
                print(content_t.shape, generated_t.shape)

                generated_t1 = stylization_network(content_t1)



                # generated_t.data.clamp_(0, 1)    # should we clamp? Tut does, example doesn't
                # generated_t1.data.clamp_(0, 1)

                # clears gradients for each iteration of backprop
                # optimizer.zero_grad()

                # calculate content losses for current and previous frame
                content_t_style_activations = spatial_loss_network(content_t)
                content_t1_style_activations = spatial_loss_network(content_t1)
                generated_t_style_activations = spatial_loss_network(generated_t)
                generated_t1_style_activations = spatial_loss_network(generated_t1)

                # [3] = last of the returned activation maps
                content_loss_t = content_loss(content_t_style_activations[3], generated_t_style_activations[3])
                content_loss_t1 = content_loss(content_t1_style_activations[3], generated_t1_style_activations[3])

                # total content loss (section 3.2.1)
                total_content_loss = content_loss_t + content_loss_t1
                total_content_loss *= content_weight
                print("content loss", total_content_loss)

                # calculate style losses for current and previous frame
                style_image_activations = spatial_loss_network(style_img)

                style_loss_t = 0
                style_loss_t1 = 0
                for i in range(len(style_image_activations)):
                    style_loss_t += style_loss(style_image_activations[i], generated_t_style_activations[i])
                    style_loss_t1 += style_loss(style_image_activations[i], generated_t1_style_activations[i])

                # total style loss (section 3.2.1)
                total_style_loss = style_loss_t + style_loss_t1
                total_style_loss *= style_weight
                print("style loss", total_style_loss)



                # regularization (TV Regularizer, section 3.2.1)
                tv_loss = tv_loss(generated_t_style_activations[3]) #???
                tv_loss *= variation_weight
                print("tv loss", tv_loss)


                # final spatial loss
                spatial_loss = total_style_loss + total_content_loss + tv_loss

                # Optical flow (Temporal Loss, section 3.2.2)
                # flow_t1, mask = opticalflow(generated_t.squeeze(0).permute(1, 2, 0).data.numpy(), generated_t1.squeeze(0).permute(1, 2, 0).data.numpy())
                #
                # temporal_loss = TemporalLoss(generated_t, flow_t1, mask)
                # temporal_loss *= temporal_weight

                # Hybrid loss and backprop
                # hybrid_loss = spatial_loss + temporal_loss
                print("Spatial loss is: ", spatial_loss)
                hybrid_loss = spatial_loss
                hybrid_loss.backward(retain_graph=True)

                optimizer.step()

                steps_completed += 1
                if steps_completed % 50 == 0:
                    print("Step {}:".format(steps_completed))
                    print('Style Loss : {:4f} Content Loss: {:4f} Temporal Loss: {:4f}'.format(
                        total_style_loss.item(), total_content_loss.item(), temporal_loss.item()))
                    print()
        # save the model parameters after training
        torch.save(stylization_network.state_dict(), model_path)


train_stylization_network(style_img)

# plt.figure()
# imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
# plt.show()

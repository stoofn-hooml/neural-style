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

import datetime

print(torch.cuda.device_count())

if (torch.cuda.is_available()):
    print("CUDA available")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CPU")
    device = torch.device("cpu")

# use shape instead of size so we can be sure that content frames will be the same
# img_shape = (360, 640) if torch.cuda.is_available() else (72, 128)
img_shape = (240, 426)
print(img_shape)

transform = transforms.Compose([
    transforms.Resize(img_shape),  # scale imported image
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))])  # transform it into a torch tensor

def normalizeTensor(tensor):
    # normalize using imagenet mean and std
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    small_tensor = tensor.clone().div_(255.0)
    return (small_tensor - mean) / std

# normalizeImg = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# resize, convert to tensor, add dimension, put on GPU if available
def transformImg(image, style=True, normalize=True):
    # fake batch dimension required to fit network's input dimensions
    image = transform(image) # adds another dimension to tensor
    if (style):
        image = image.unsqueeze(0) # adds another dimension to style tensor
    # if (normalize):
    #     image = normalizeImg(image)
    return image.to(device, torch.float)


style_img = transformImg(Image.open("./images/starry.jpg"), style=True, normalize=True)
content_path = './content_videos'
unloader = transforms.ToPILImage()  # reconvert into PIL image

# plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are update

# plt.figure()
# imshow(style_img)


model_save_path = "./models/model" + str(datetime.datetime.now()).strip('/') + ".pth"
model_load_path = "./models/model.pth"

load_model = True

use_temporal_loss = False

# hyperperameter defaults (specified in section 4.1)
default_content_weight = 1
default_style_weight = 10
default_temporal_weight = 10000
default_variation_weight = .001

default_epochs = 2

# style_loss_layers = [1]

def saveModel(stylization_network, optimizer, steps_completed):
    # save the state dict in evaluation mode
    stylization_network.eval()
    print("Saving model to:", model_save_path)
    torch.save({
        'frames': steps_completed,
        'model_state_dict': stylization_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, model_save_path)
    # return the model to training mode
    stylization_network.to(device).train()


def train_stylization_network(style_img, epochs=default_epochs,
                       content_weight=default_content_weight,
                       style_weight=default_style_weight,
                       temporal_weight=default_temporal_weight,
                       variation_weight=default_variation_weight):
    """Run the style transfer."""
    print('Building the style transfer model...')

    # initialize starting values/networks
    stylization_network = StylizationNetwork().to(device)
    # using default learning rate 0.001
    optimizer = optim.Adam(stylization_network.parameters())

    previous_steps = 0
    if (load_model):
        print("Loading: ", model_load_path)
        checkpoint = torch.load(model_load_path, map_location='cpu')
        stylization_network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        previous_steps = checkpoint['frames']

    # initialize spatial loss network and hybrid loss modules
    spatial_loss_network = SpatialLossNetwork(device).to(device)

    content_loss = ContentLoss(device).to(device)
    style_loss = StyleLoss(device).to(device)
    tv = TVLoss(device).to(device)
    temporal = TemporalLoss(device).to(device)

    # calculate style activations

    print('Training...')
    steps_completed = previous_steps

    # get loader of video
    video_loader = get_loader(1, content_path, transformImg)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        stylization_network.train()

        # video_loader is an iterator so it must be accessed with enumerate()
        # each element of enumerate(video_loader) is a list of frames (a video)
        for j, frames in enumerate(video_loader):
            print("Video", j)
            # loop through all the frames for each video
            for i in range(1, len(frames)):
                # clears gradients for each iteration of backprop
                optimizer.zero_grad()

                content_t = frames[i];      # current frame
                # content_t1 = frames[i-1];   # previous frame
                # imshow(content_t)

                generated_t = stylization_network(content_t).to(device)
                # generated_t1 = stylization_network(content_t1).to(device)

                content_t = normalizeTensor(content_t)
                # content_t1 = normalizeTensor(content_t1)
                generated_t = normalizeTensor(generated_t)
                # generated_t1 = normalizeTensor(generated_t1)

                # don't clamp! Then it's not the output of the stylization network being optimized
                # generated_t.data.clamp_(0, 1)    # should we clamp? Tut does, example doesn't
                # generated_t1.data.clamp_(0, 1)

                # calculate content losses for current and previous frame
                content_t_style_activations = spatial_loss_network(content_t)
                # content_t1_style_activations = spatial_loss_network(content_t1)
                generated_t_style_activations = spatial_loss_network(generated_t)
                # generated_t1_style_activations = spatial_loss_network(generated_t1)

                # [3] = last of the returned activation maps
                content_loss_t = content_loss(content_t_style_activations[3], generated_t_style_activations[3]).to(device)
                # content_loss_t1 = content_loss(content_t1_style_activations[3], generated_t1_style_activations[3]).to(device)

                # total content loss (section 3.2.1)
                # total_content_loss = (content_loss_t + content_loss_t1) * content_weight
                total_content_loss = content_loss_t
                total_content_loss *= content_weight

                style_image_activations = spatial_loss_network(normalizeTensor(style_img))

                style_loss_t = 0
                # style_loss_t1 = 0
                for i in range(len(style_image_activations)):
                    style_loss_t += style_loss(style_image_activations[i], generated_t_style_activations[i]).to(device)
                    # style_loss_t1 += style_loss(style_image_activations[i], generated_t1_style_activations[i]).to(device)

                # total style loss (section 3.2.1)
                # total_style_loss = (style_loss_t + style_loss_t1) * style_weight
                total_style_loss = style_loss_t
                total_style_loss *= style_weight

                # regularization (TV Regularizer, section 3.2.1)
                # tv_loss = tv(generated_t_style_activations[3]).to(device) #???
                # tv_loss *= variation_weight
                # tv_loss = 0

                # final spatial loss
                # spatial_loss = total_style_loss + total_content_loss + tv_loss
                spatial_loss = total_style_loss + total_content_loss


                # Optical flow (Temporal Loss, section 3.2.2)
                # .squeeze(0).permute(1, 2, 0)
                # if use_temporal_loss:
                #     optical_t = generated_t.squeeze(0).permute(1, 2, 0).cpu().data.numpy()
                #     optical_t1 = generated_t1.squeeze(0).permute(1, 2, 0).cpu().data.numpy()
                #     flow_t1, mask = opticalflow(optical_t, optical_t1)
                #
                #     temporal_loss = temporal(generated_t, generated_t1, flow_t1, mask)
                #     temporal_loss *= temporal_weight
                #
                #     hybrid_loss = spatial_loss + temporal_loss
                # else:
                hybrid_loss = spatial_loss

                # calculate gradients for backprop
                hybrid_loss.backward()

                # optimize parameters of stylization network from backprop
                optimizer.step()

                steps_completed += 1
                if steps_completed % 50 == 0:
                    print("Frames completed {}:".format(steps_completed))
                    print('Hybrid loss: {:4f}\nStyle Loss : {:4f} Content Loss: {:8f}'.format(hybrid_loss.item(),
                        total_style_loss.item(), total_content_loss.item())) #   TV Loss: {:4f} tv_loss.item()
                    print()

            saveModel(stylization_network, optimizer, steps_completed)
        # save the model parameters after training
        # torch.save(stylization_network.state_dict(), model_path)

if __name__ == "__main__":
    train_stylization_network(style_img)

# plt.figure()
# imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
# plt.show()

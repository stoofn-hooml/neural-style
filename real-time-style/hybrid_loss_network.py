import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

""" code skeleton taken from Pytorch tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalization(nn.Module):
    """ module to normalize input image so we can easily put it in a nn.Sequential """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.

        # specified values for VGG 19
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, content_activation, generated_activation):
        return (generated_activation - self.mean) / self.std

class ContentLoss(nn.Module):
    """ computes loss between content image and generated using MSE for spatial loss"""

    def __init__(self, device):
        super(ContentLoss, self).__init__()
        self.loss = nn.MSELoss().to(device)

    def forward(self, content_activation, generated_activation):
        # inputs are the respective feature maps at ReLU_10 (final loss layer)
        b, c, h, w = content_activation.shape
        return (1/(c*h*w) * self.loss(content_activation, generated_activation))

def GramMatrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d).cuda()

class StyleLoss(nn.Module):
    """ computes loss between style image and generated image using Gram matrix for spatial loss """

    def __init__(self, device):
        super(StyleLoss, self).__init__()
        self.loss = nn.MSELoss().to(device)

    def forward(self, style_activation, generated_activation):
        style_gram = GramMatrix(style_activation)
        generated_gram = GramMatrix(generated_activation)

        num_channels = generated_activation.shape[3]
        return (1 / num_channels ** 2) * self.loss(style_gram, generated_gram)

class TVLoss(nn.Module):
    """ TV regularization factor for spatial loss (Huang section 3.2.1) """
    def __init__(self, device):
        super(TVLoss, self).__init__()
        self.device = device

    def forward(self, generated_activation): #calculates overall pixel smoothness for handling checkerboard artifacts
        # source:  https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
        return torch.sum(torch.abs(generated_activation[:, :, :, :-1] - generated_activation[:, :, :, 1:])) + torch.sum(torch.abs(generated_activation[:, :, :-1, :] - generated_activation[:, :, 1:, :]))


class TemporalLoss(nn.Module):
    """
    This module is unused in our final implementation
    computes loss between consecutive generated frame
    generated_t: frame t
    flow_t1: optical flow(frame t-1)
    mask: confidence mask of optical flow
    """
    def __init__(self, device):
        super(TemporalLoss, self).__init__()
        self.loss = nn.MSELoss().to(device)

    def forward(self, generated_t, generated_t1, flow_t1, mask):
        b, c, h, w = generated_t.shape
        D = h * w * c
        warped = nn.grid_sample(generated_t1, )
        return ((1 / D) * mask * self.loss(generated_t, warp_image(generated_t1, flow_t1)))


class SpatialLossNetwork(nn.Module):
    def __init__(self, device):
        super(SpatialLossNetwork, self).__init__()
        features = list(models.vgg19(pretrained=True).features.to(device))[:23]
        self.features = nn.ModuleList(features).eval().to(device)
        self.device = device

    # source: https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x).to(self.device)
            # indices of style layers (found in section 3.2.1)
            if ii in {3, 8, 13, 22}:
                results.append(x)
        return results

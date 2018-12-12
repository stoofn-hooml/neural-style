# neural-style

This repository contains two separate systems for neural style transfer: real-time-style and multi-style. The dependencies and instructions for each are detailed below.

# Real-time-style
This system performs neural style transfer for videos in real-time using a pre-trained feedforward convolutional neural network. The architecture is based on the paper ["Real-Time Neural Style Transfer for Videos"](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf) by Huang et al. It is built to train a model that will stylize arbitrary content based on a single style image.

## Dependencies
The system was written in Python, and it uses the  [Pytorch](https://pytorch.org/) library for deep learning. Additionally, it requires several python packages that can be installed via pip: torchvision, PIL, matplotlib, and cv2.

It is advised to train the model with a GPU, since the process is very slow on CPU. GPU capability requires additional dependencies. For an NVIDIA based GPU, they are CUDA 6.5+ and the cudnn backend for pytorch. We tested the code on an NVIDIA Titan Xp GPU, running on a Linux server with CUDA 9.0.

## Training
Before training, you must download a dataset of content videos. We have included a videocrawler.py script, which scrapes 100 random videos from [Videvo.net](https://www.videvo.net/). To use this script, make sure to set a proper save path at the top of code.

In `train_stylization_network.py`, set the proper paths for the `content_path` of these downloaded videos and the `style_img` to train the stylization (we have included example style images in the images folder). To train the stylization network to stylize a video with this style, run the script train_stylization_network with the command:
```
python3 stylization_network.py
```
The script will save checkpoints of the model after training with the frames of each video. These .pth files will be located in the models folder. To resume training with a previous checkpoint, set `model_load_path` to the appropriate .pth file and set `load_model` to True.

## Stylizing
Once a model has been trained, you can test its stylization with the `stylize_video.py` script. Set `video_path` to the video to stylize in real-time (we have included an example video in the videos_to_stylize folder). Then set `model_path` to the .pth checkpoint file that you would like to test (we have included example models in the folder models). Run the script with the command:
```
python3 stylize_video.py
```
The script will display the result of stylizing each frame of the input video on loop.

# Multi-style
This is a neural style transfer system that allows _style videos_. It takes as input a content video and style video and generates an output video with a dynamic style. To stylize each frame, it uses the neural-style.lua script created by Johnson et al., found from [this repository](https://github.com/jcjohnson/neural-style).

<img src="multi-style/example_outputs/lionloop.gif" width="550">

## Dependencies
The script is written in Python. It requires [Pytorch](https://pytorch.org/), as well as the python packages PIL, plt, and cv2. It calls Johnson's stylization script internally, which requires [torch7](https://github.com/torch/torch7) and [loadcaffe](https://github.com/szagoruyko/loadcaffe).
To stitch the resulting stylized frames into a video, the command line tool [ffmpeg](https://www.ffmpeg.org/) is required.

## Stylizing
First, you must download the VGG-19 model required for image stylization by running the command:
```
sh models/download_models.sh
```
Set the desired paths for `content_video` and `style_video` at the top of `multi_style_transfer.py`. We have included several example videos in the folders content_videos and style_videos. Then run the program with the command:
```
python3 multi_style_transfer.py
```
In an output folder labeled with the current date and time, the program will output the content frames, style frames, resulting stylized frames, and final stylized video.

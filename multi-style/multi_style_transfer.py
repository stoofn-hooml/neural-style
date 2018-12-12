import os
import sys
import subprocess
import shlex
import datetime

import torch
from torchvision import transforms, utils

from PIL import Image
import matplotlib.pyplot as plt
import cv2

content_video = "./content_videos/lion.mpeg"
style_video = "./style_videos/paint.mpeg"

if (torch.cuda.is_available()):
    print("CUDA available")
    device = torch.device("cuda")
    print("Using GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    img_shape = (360, 640)
else:
    print("Using CPU")
    device = torch.device("cpu")
    img_shape = (240, 426)


loader = transforms.Compose([
    transforms.Resize(img_shape),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image):
    # image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def get_video_frames(video_path):
    video = cv2.VideoCapture(video_path)

    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)

    framecount = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if (ret):
            framecount += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            transformed_frame = image_loader(pil_frame)
            frames.append(transformed_frame)
        else:
            video.release()
    return fps, frames

def make_dirs(save_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "/content", exist_ok=True)
    os.makedirs(save_path + "/style", exist_ok=True)
    os.makedirs(save_path + "/stylized", exist_ok=True)

# helper function to get full paths of videos in each directory
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def get_frame_paths(dir):
    paths = listdir_fullpath(dir)
    paths.sort()
    return paths

def create_output_video(fps, frame_path):
    subprocess.call(shlex.split('ffmpeg -framerate {} -start_number 0 -i {} -c:v libx264 -pix_fmt yuv420p out.mp4'.format(fps, frame_path)))

def run_multi_style_transfer():
    print("Loading frames...")
    content_fps, content_frames = get_video_frames(content_video)
    style_fps, style_frames = get_video_frames(style_video)


    now = datetime.datetime.now()
    save_path = "./output." + str(now.day) + "." + str(now.hour) + "." + str(now.minute)
    print("Saving output to: ", save_path)

    make_dirs(save_path)

    print("Saving frames...")
    for i, frame in enumerate(content_frames):
        utils.save_image(frame, save_path + "/content/content_" + str(i).zfill(4) + ".png")

    for i, frame in enumerate(style_frames):
        utils.save_image(frame, save_path + "/style/style_" + str(i).zfill(4) + ".png")


    stylized_frames = []

    content_paths = get_frame_paths(save_path + "/content/")
    style_paths = get_frame_paths(save_path + "/style/")

    # content_frames_per_style_frame
    # style_frame_skip
    for i in range(len(content_paths))[:2]:
        print("Stylizing frame "+str(i))

        content_path = content_paths[i]
        style_path = style_paths[i % len(style_paths)]

        stylized_path = save_path + "/stylized/stylized_" + str(i).zfill(4) + ".png"

        subprocess.call(shlex.split("th neural_style.lua -style_image {} -content_image {} -output_image {} -save_iter 0 -num_iterations 750".format(style_path, content_path, stylized_path)))

    create_output_video(content_fps, save_path + "/stylized/stylized_%04.png")


if __name__ == "__main__":
    run_multi_style_transfer()

import os
import cv2
import torch.utils.data as data

from PIL import Image

from torchvision import datasets, transforms

""" modified from curaai00 on Github: https://github.com/curaai00/RT-StyleTransfer-forVideo/blob/master/opticalflow.py """

class Dataset(data.Dataset):
    def __init__(self, data_path, transformImg):
        self.data_path = data_path
        self.transformImg = transformImg
        self.video_list = [file for file in os.listdir(data_path) if file[0] != "."]
        self.video_list.sort()

    def __getitem__(self, i):
        video = cv2.VideoCapture(os.path.join(self.data_path, self.video_list[i]))

        frames = []

        framecount = 0
        while (video.isOpened()):
            ret, frame = video.read()

            if (ret):
                framecount += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # resize, convert to tensor, add dimension, put on GPU if available
                transformed_frame = self.transformImg(pil_frame, style=False, normalize=True)
                frames.append(transformed_frame)
            else:
                video.release()

        print("Loaded", self.video_list[i])
        return frames

    def __len__(self):
        return len(self.video_list)


def get_loader(batch_size, data_path, transformImg, shuffle=True):
    dataset = Dataset(data_path, transformImg)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return loader

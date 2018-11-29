import os
import cv2
import torch
import torch.utils.data as data

from PIL import Image

from torchvision import datasets, transforms

""" taken from curaai00 on Github: https://github.com/curaai00/RT-StyleTransfer-forVideo/blob/master/opticalflow.py """

class Dataset(data.Dataset):
    def __init__(self, data_path, transform_img):
        self.data_path = data_path
        self.transform_img = transform_img
        self.video_list = os.listdir(data_path)

    def __getitem__(self, i):
        video = cv2.VideoCapture(os.path.join(self.data_path, self.video_list[i]))

        print(os.path.join(self.data_path, self.video_list[i]))
        frames = []

        framecount = 0
        while (video.isOpened()):
            ret, frame = video.read()

            if (ret):
                framecount += 1
                print("frame ", framecount)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # resize, convert to tensor, add dimension, put on GPU if available
                # rgb_frame = self.transform_img(rgb_frame)
                transformed_frame = self.transform_img(pil_frame, False)
                frames.append(transformed_frame)
                # print("appended")
            else:
                video.release()

        return frames

    def __len__(self):
        return len(self.video_list)


def get_loader(batch_size, data_path, transform_img, shuffle=True):
    dataset = Dataset(data_path, transform_img)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return loader

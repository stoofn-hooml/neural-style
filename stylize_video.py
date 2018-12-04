import cv2
from stylization_network import StylizationNetwork
from train_stylization_network import transformImg, imshow
import torchvision.transforms as transforms
import torch
from PIL import Image
import re

import matplotlib.pyplot as plt

from torch.utils.serialization import load_lua

from transformer_net import TransformerNet

# video_path = './videos_to_stylize/fulldogs.mov'
video_path = './videos_to_stylize/29_34.mp4'

video = cv2.VideoCapture(video_path)

# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
model_path = "./models/model.pth"
stylization_network = StylizationNetwork()
#
checkpoint = torch.load(model_path, map_location='cpu')
stylization_network.load_state_dict(checkpoint['model_state_dict'])
stylization_network.eval()

# stylization_network = load_lua('./models/starry_night.t7')

# stylization_network = TransformerNet()
# state_dict = torch.load('./models/mosaic.pth')
# # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
# for k in list(state_dict.keys()):
#     if re.search(r'in\d+\.running_(mean|var)$', k):
#         del state_dict[k]
# stylization_network.load_state_dict(state_dict)

# stylization_network.eval()


fps = video.get(cv2.CAP_PROP_FPS)
wait_time = int(1/fps*360)

if (not video.isOpened()):
    print("Error loading ", video_path)

frame_count = 0

scaleTensor = transforms.Lambda(lambda x: x.mul(255))

while(True):
    ret, frame = video.read()
    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)

    transformed_frame = transformImg(pil_frame, style=True)
    # print(transformed_frame)
    # imshow(transformed_frame)

    # Other guy
    # transformed_frame = scaleTensor(transformed_frame)

    stylized_frame = stylization_network(transformed_frame)

    # stylized_frame = stylized_frame.clamp(0, 1)
    # print(stylized_frame)


    # stylized_frame = stylized_frame[0].detach().clamp(0, 255).numpy()
    # stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    # print(stylized_frame)
    # cv2.imshow("Stylized", stylized_frame)

    # print(stylized_frame)
    imshow(stylized_frame, title='Stylized')

    cv2.waitKey(wait_time)

    if (frame_count == video.get(cv2.CAP_PROP_FRAME_COUNT)-50):
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        frame_count = 0 #Or whatever as long as it is the same as next line

video.release()
cv2.destroyAllWindows()

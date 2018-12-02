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

video_path = './videos_to_stylize/fulldogs.mov'

video = cv2.VideoCapture(video_path)

# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
# model_path = "./models/model.pth"
# stylization_network = StylizationNetwork()
# stylization_network.load_state_dict(torch.load(model_path, map_location='cpu'))
# stylization_network.eval()

# stylization_network = load_lua('./models/starry_night.t7')

stylization_network = TransformerNet()
state_dict = torch.load('./models/mosaic.pth')
# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
stylization_network.load_state_dict(state_dict)

# stylization_network = torch.load('./models/starry_night.t7')

fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
wait_time = int(1/fps*360)

if (not video.isOpened()):
    print("Error loading ", video_path)

frame_count = 0

content_transform = transforms.Compose([
    transforms.Resize((240, 426)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

while(True):
    ret, frame = video.read()
    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)

    # print(pil_frame)
    transformed_frame = content_transform(pil_frame)
    transformed_frame = transformed_frame.unsqueeze(0)

    # resize, convert to tensor, add dimension, put on GPU if available
    # rgb_frame = self.transformImg(rgb_frame)
    # transformed_frame = transformImg(pil_frame, True, True)

    output = stylization_network(transformed_frame).cpu()
    # stylized_frame.data.clamp_(0, 1)

    # print(stylized_frame, stylized_frame.shape)
    # [1, 3, 240, 428]
    stylized_frame = output[0].detach().clamp(0, 255).numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    # stylized_frame = Image.fromarray(stylized_frame)

    # mean: (0.485, 0.456, 0.406), std: (0.229, 0.224, 0.225)

    # stylized_frame[0] = stylized_frame[0]*0.229 + 0.485
    # stylized_frame[1] = stylized_frame[1]*0.224 + 0.456
    # stylized_frame[2] = stylized_frame[2]*0.225 + 0.406

    # stylized_frame[0] = stylized_frame[0]+ 0.485
    # stylized_frame[1] = stylized_frame[1] + 0.456
    # stylized_frame[2] = stylized_frame[2] + 0.406
    # stylized_frame *= 255

    # stylized_frame.data.clamp_(0, 1)

    # stylized_frame = stylized_frame.permute(1, 2, 0).cpu().data.numpy()
    # print(stylized_frame, stylized_frame.shape)
    # [240, 428, 3]

    cv2.imshow("Stylized", stylized_frame)
    # stylized_frame = stylized_frame.reshape((3, output.shape[2], output.shape[3]))
	# stylized_frame[0] += 103.939
	# stylized_frame[1] += 116.779
	# stylized_frame[2] += 123.680
	# stylized_frame /= 255.0
	# stylized_frame = stylized_frame.transpose(1, 2, 0)

    # print(stylized_frame, stylized_frame.shape)

    # plt.figure()
    # imshow(stylized_frame, title='Stylized')

    # plt.figure()
    # imshow(transformed_frame, title='Normal')

    # cv2.imshow('Frame', frame)


    # Press Q on keyboard to  exit
    # print(wait_time)
    cv2.waitKey(wait_time)
    # if cv2.waitKey(wait_time) & 0xFF == ord('q'):
    #   break

    # print (video.get(cv2.CAP_PROP_FRAME_COUNT)-video.get(cv2.CAP_PROP_FRAME_COUNT)/5)
    # print (frame_count)
    if (frame_count == video.get(cv2.CAP_PROP_FRAME_COUNT)-50):
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        frame_count = 0 #Or whatever as long as it is the same as next line

video.release()
cv2.destroyAllWindows()

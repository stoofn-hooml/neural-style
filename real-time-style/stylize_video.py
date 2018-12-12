import torch
import torchvision.transforms as transforms

import cv2
from PIL import Image

from stylization_network import StylizationNetwork
from train_stylization_network import imshow

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

video_path = './videos_to_stylize/dogs.mov'
video = cv2.VideoCapture(video_path)

# source: https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
model_path = "./models/picasso.pth"

stylization_network = StylizationNetwork()

checkpoint = torch.load(model_path, map_location='cpu')
stylization_network.load_state_dict(checkpoint['model_state_dict'])
stylization_network.eval()

img_shape = (360, 640)

transform = transforms.Compose([
    transforms.Resize(img_shape),  # scale imported image
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))])  # transform it into a torch tensor

# resize, convert to tensor, add dimension, put on GPU if available
def transformImg(image, style=True, normalize=True):
    # fake batch dimension required to fit network's input dimensions
    image = transform(image)
    if (style):
        image = image.unsqueeze(0) # adds another dimension to style tensor
    return image.to(device, torch.float)


fps = video.get(cv2.CAP_PROP_FPS)
wait_time = int(1/fps*360)

if (not video.isOpened()):
    print("Error loading ", video_path)

frame_count = 0

while(True):
    ret, frame = video.read()
    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)
    transformed_frame = transformImg(pil_frame, style=True)

    stylized_frame = stylization_network(transformed_frame)
    stylized_frame = stylized_frame.clamp(0, 1) # clamp the image to standard values

    imshow(stylized_frame, title='Stylized')

    cv2.waitKey(wait_time)

    if (frame_count == video.get(cv2.CAP_PROP_FRAME_COUNT)-50):
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        frame_count = 0

video.release()
cv2.destroyAllWindows()

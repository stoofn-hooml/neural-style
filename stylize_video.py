import cv2

video_path = './videos_to_stylize/fulldogs.mov'

video = cv2.VideoCapture(video_path)

# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
model_path = model_path = "./models/model.pth"
# stylization_network = StylizationNetwork()
# stylization_network.load_state_dict(torch.load(model_path))

fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
wait_time = int(1/fps*360)
# wait_time = 1
print(wait_time)

if (not video.isOpened()):
    print("Error loading ", video_path)

frame_count = 0

while(True):
    ret, frame = video.read()
    frame_count += 1

    frame = cv2.resize(frame, (640, 360))
    # stylized_frame = stylization_network(frame)

    cv2.imshow('Frame', frame)


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

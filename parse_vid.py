import cv2
import pandas as pd
import torch
import random
from tqdm import tqdm
from model import RacingNet
from torchvision import transforms
from PIL import Image

video_id = 9
video_path = "videos/20230627_171910.mp4"
cap = cv2.VideoCapture(video_path)
START_MIN = 0
SELECT_PROB = 1/30.0  # Probability to select each frame
FLIP_HORIZONTAL = False

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = START_MIN * 60 * int(cap.get(cv2.CAP_PROP_FPS))
total_frames -= start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

pbar = tqdm(total=total_frames)

frame_count = 0
frames_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        # Select frame with probability 1/n
        if random.random() < SELECT_PROB:
            frame_file = f"data/frames/frame_{video_id:02d}_{str(frame_count + start_frame).zfill(5)}.jpg"
            cv2.imwrite(frame_file, frame)

            frame = Image.fromarray(frame)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            frames_data.append({'frame_file': frame_file, 'timestamp': timestamp})

        frame_count += 1
        pbar.update(1)
    else:
        break

cap.release()
pbar.close()

df = pd.DataFrame(frames_data)
df.to_csv(f"data/frames_data_{video_id:02d}.csv")

print('Processing done.')

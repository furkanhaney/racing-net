import cv2
import pandas as pd
from tqdm import tqdm


# Define the video file
video_path = "videos/gt7_vid1.mp4"
cap = cv2.VideoCapture(video_path)

START_MIN = 6  # Start from 0th minute
FREQ = 240  # Frequency to save frames

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Compute the starting frame
start_frame = START_MIN * 60 * int(cap.get(cv2.CAP_PROP_FPS))

# Adjust total frames
total_frames = total_frames - start_frame

frame_count = 0
frames_data = []

# Set the video capture to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create a progress bar
pbar = tqdm(total=total_frames)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        if frame_count % FREQ == 0:  # Take every FREQth frame
            # Frame file path
            frame_file = f"data/frames/frame_00_{str(frame_count + start_frame).zfill(5)}.jpg"

            # Write the frame to a JPG file
            cv2.imwrite(frame_file, frame)

            # Get the timestamp of the current frame (in seconds)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Save the frame data
            frames_data.append({
                'frame_file': frame_file,
                'timestamp': timestamp,
            })

        frame_count += 1
        pbar.update(1)
    else:
        break

# When everything is done, release the capture
cap.release()
pbar.close()

# Convert the list to a pandas DataFrame and then to CSV
df = pd.DataFrame(frames_data)
df.to_csv("data/frames_data.csv")

print('Processing done.')

import os
import shutil
import pandas as pd

# Source paths
source_frames_dir = 'data/frames'
source_csv_file = 'data/frames_data.csv'

# Destination paths
dest_dir = 'dataset/'
dest_frames_dir = os.path.join(dest_dir, 'frames')
dest_csv_file = os.path.join(dest_dir, 'frames_data.csv')

# Create destination directories if they do not exist
os.makedirs(dest_frames_dir, exist_ok=True)

# Load the frame data
df = pd.read_csv(source_csv_file)

# Copy only the frames that were used
for _, row in df.iterrows():
    # Get the source frame file
    source_frame_file = row['frame_file']

    # Construct the destination frame file path
    dest_frame_file = os.path.join(dest_frames_dir, os.path.basename(source_frame_file))

    # Copy the frame file
    shutil.copy2(source_frame_file, dest_frame_file)

# Copy the csv file
shutil.copy2(source_csv_file, dest_csv_file)

print('Dataset creation done.')

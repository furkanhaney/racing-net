import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Set parameters
image_width = 512
image_height = 512
data_file = "data/data.xlsx"
dest_dir = "dataset/frames"

df = pd.read_excel(data_file)
df = df[df["speed"].notna()]

os.makedirs(dest_dir, exist_ok=True)

new_images = 0

for index, row in tqdm(df.iterrows(), total=len(df)):
    dest_frame_file = os.path.join(dest_dir, os.path.basename(row["frame_file"]))

    if os.path.exists(dest_frame_file):
        continue
    img = Image.open(row["frame_file"])
    img_resized = img.resize((image_width, image_height))
    img_resized.save(dest_frame_file)
    new_images += 1

print(f"Added {new_images} new images.")

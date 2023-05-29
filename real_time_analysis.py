# real_time_analysis.py
import torch
import pandas as pd
from model import RacingNet
from data import RacingDataset
from PIL import Image
import numpy as np
import time


def analyze_real_time_data(model, device, data_stream):
    """Analyze real-time data.

    Args:
        model (RacingNet): Trained model.
        device (torch.device): Device (GPU/CPU) where the model is.
        data_stream (generator): Generator that streams the data.

    """
    model.eval()
    with torch.no_grad():
        for frame_path, csv_data in data_stream:
            # Load the image
            with Image.open(frame_path) as img:
                image = img.convert('RGB').resize((224, 224), Image.ANTIALIAS)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).unsqueeze(0).to(device)

            # Load the racing data
            racing_data = pd.read_csv(csv_data)
            in_race = torch.tensor(racing_data['in_race']).unsqueeze(0).to(device)
            speed = torch.tensor(racing_data['speed']).unsqueeze(0).to(device)
            position = torch.tensor(racing_data['position']).unsqueeze(0).to(device)
            lap = torch.tensor(racing_data['lap']).unsqueeze(0).to(device)
            gear = torch.tensor(racing_data['gear']).unsqueeze(0).to(device)

            # Predict
            pred_in_race, pred_speed, pred_position, pred_gear, pred_lap = model(image)

            # Print predictions
            print("Predicted in_race: ", pred_in_race.item())
            print("Predicted speed: ", pred_speed.item())
            print("Predicted position: ", pred_position.argmax(dim=1).item())
            print("Predicted gear: ", pred_gear.argmax(dim=1).item())
            print("Predicted lap: ", pred_lap.argmax(dim=1).item())

            # Compare with actual data
            print("Actual in_race: ", in_race.item())
            print("Actual speed: ", speed.item())
            print("Actual position: ", position.item())
            print("Actual gear: ", gear.item())
            print("Actual lap: ", lap.item())

            # Sleep for a while before the next prediction
            time.sleep(0.5)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RacingNet().to(device)

    # Load trained model weights
    # model.load_state_dict(torch.load('model_weights.pth'))

    # Define a generator that streams the data
    # This is a placeholder for the actual data stream generator
    def data_stream():
        while True:
            yield 'frame.jpg', 'data.csv'

    analyze_real_time_data(model, device, data_stream())

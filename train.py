import torch
import torch.nn as nn
from model import RacingNet
from data import get_loaders
from eval import calculate_speed_error, calculate_accuracy


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    total_loss = 0
    for batch in data_loader:
        images, in_race, true_speed, true_position, true_lap, true_gear = batch
        images = images.to(device)
        in_race = in_race.to(device)
        true_speed = true_speed.to(device)
        true_position = true_position.to(device)
        true_lap = true_lap.to(device)
        true_gear = true_gear.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        pred_in_race, pred_speed, pred_position, pred_gear, pred_lap = model(images)

        # Calculate loss
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()

        in_race_loss = bce_loss(pred_in_race, in_race[:, None].float())
        speed_loss = mse_loss(pred_speed.squeeze(), true_speed.float())
        position_loss = ce_loss(pred_position, true_position.long())
        gear_loss = ce_loss(pred_gear, true_gear.long())
        lap_loss = ce_loss(pred_lap, true_lap.long())

        total_loss = in_race_loss + speed_loss + position_loss + gear_loss + lap_loss

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

    return total_loss.item()


def evaluate_one_epoch(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        total_speed_error = 0.0
        total_gear_accuracy = 0.0
        total_position_accuracy = 0.0
        total_lap_accuracy = 0.0
        num_samples = 0

        for batch in data_loader:
            images, in_race, true_speed, true_position, true_lap, true_gear = batch
            images = images.to(device)

            # Forward pass
            pred_in_race, pred_speed, pred_position, pred_gear, pred_lap = model(images)
            # Calculate mean mph error for speed
            speed_error = calculate_speed_error(pred_speed, true_speed)
            total_speed_error += speed_error.item()

            # Convert predicted gear, position and lap to integer
            pred_gear = torch.argmax(pred_gear, dim=1)
            pred_position = torch.argmax(pred_position, dim=1)
            pred_lap = torch.argmax(pred_lap, dim=1)

            # Calculate accuracy for gear, position, and laps
            gear_accuracy = calculate_accuracy(pred_gear, true_gear)
            total_gear_accuracy += gear_accuracy
            position_accuracy = calculate_accuracy(pred_position, true_position)
            total_position_accuracy += position_accuracy
            lap_accuracy = calculate_accuracy(pred_lap, true_lap)
            total_lap_accuracy += lap_accuracy

            num_samples += images.size(0)

        # Calculate average metrics
        avg_speed_error = total_speed_error / num_samples
        avg_gear_accuracy = total_gear_accuracy / num_samples
        avg_position_accuracy = total_position_accuracy / num_samples
        avg_lap_accuracy = total_lap_accuracy / num_samples

    return avg_speed_error, avg_gear_accuracy, avg_position_accuracy, avg_lap_accuracy


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set batch size
    batch_size = 16

    # Load training and validation data
    train_loader, val_loader = get_loaders(batch_size)

    # Create an instance of the model
    model = RacingNet().to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the number of epochs to train
    num_epochs = 10

    for epoch in range(num_epochs):
        # Training
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss:.4f}")

        # Validation
        speed_error, gear_accuracy, position_accuracy, lap_accuracy = evaluate_one_epoch(model, val_loader, device)
        print(f"Validation Speed Error: {speed_error:.2f}, Gear Accuracy: {gear_accuracy:.2%}, Position Accuracy: {position_accuracy:.2%}, Lap Accuracy: {lap_accuracy:.2%}")

    # Save the model weights
    torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == "__main__":
    main()

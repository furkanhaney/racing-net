import torch
from model import RacingNet
from data import get_loaders


def calculate_speed_error(pred_speed, true_speed):
    """
    Calculate mean mph error for speed.
    Args:
        pred_speed (torch.Tensor): predicted speed by the model.
        true_speed (torch.Tensor): actual speed from the dataset.
    Returns:
        torch.Tensor: mean speed error.
    """
    error = torch.abs(pred_speed[:, 0] - true_speed) * 500.0
    mean_error = torch.sum(error)
    return mean_error


def calculate_accuracy(pred_labels, true_labels):
    """
    Calculate accuracy for categorical labels.
    Args:
        pred_labels (torch.Tensor): predicted labels by the model.
        true_labels (torch.Tensor): actual labels from the dataset.
    Returns:
        float: accuracy.
    """
    correct = torch.sum(pred_labels == true_labels).item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set batch size
    batch_size = 16

    # Load validation data
    _, val_loader = get_loaders(batch_size)

    # Create an instance of the model
    model = RacingNet().to(device)

    # Load trained model weights
    # model.load_state_dict(torch.load('model_weights.pth'))

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_speed_error = 0.0
        total_gear_accuracy = 0.0
        total_position_accuracy = 0.0
        total_lap_accuracy = 0.0
        num_samples = 0

        for batch in val_loader:
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

    # Print the results
    print(f"Speed Error (mean mph): {avg_speed_error:.2f}")
    print(f"Gear Accuracy: {avg_gear_accuracy:.2%}")
    print(f"Position Accuracy: {avg_position_accuracy:.2%}")
    print(f"Lap Accuracy: {avg_lap_accuracy:.2%}")


if __name__ == "__main__":
    main()

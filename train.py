import time
import torch
import torch.nn as nn
import os
import csv
from tqdm import tqdm
from model import RacingNet
from data import get_loaders
from eval import calculate_speed_error, calculate_accuracy

EXPERIMENT_NAME = "test-run-2"
NUM_ITERS = int(1e4)
EVAL_ITERS = 100
CSV_FILE_PATH = 'data/dataset.csv'
FRAMES_DIR = 'dataset/frames'
BATCH_SIZE = 32
EVAL_BATCHES = 10
FREEZE_BACKBONE = True
DECAY = 1 - 1.0 / 64
LEARNING_RATE = 1e-4

if os.path.exists(f'experiments/{EXPERIMENT_NAME}'):
    print("Please choose a new experiment name. This one already exists.")
    exit()


def evaluate_model(model, val_loader, device):
    start = time.time()
    model.eval()
    batch_count = 0
    eval_size = 0
    total_loss = 0
    total_speed_error = 0
    total_gear_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images, in_race, true_speed, true_position, true_lap, true_gear = batch
            images, in_race, true_speed, true_position, true_lap, true_gear = \
                images.to(device), in_race.to(device), true_speed.to(device), \
                    true_position.to(device), true_lap.to(device), true_gear.to(device)

            pred_in_race, pred_speed, pred_position, pred_gear, pred_lap = model(images)

            # Calculate loss and metrics
            speed_loss = nn.MSELoss()(pred_speed.squeeze(), true_speed.float())
            gear_loss = nn.CrossEntropyLoss()(pred_gear, true_gear.long())
            total_loss = 10 * speed_loss + gear_loss
            speed_error = calculate_speed_error(pred_speed, true_speed)
            gear_accuracy = calculate_accuracy(pred_gear, true_gear)

            # Accumulate metrics
            total_loss += total_loss.item()
            total_speed_error += speed_error.item()
            total_gear_accuracy += gear_accuracy
            total_samples += images.shape[0]

            batch_count += 1
            eval_size += images.shape[0]
            if batch_count == EVAL_BATCHES:
                break

    # Return the average loss and metrics
    return total_loss / total_samples, total_speed_error / total_samples, total_gear_accuracy / total_samples, time.time() - start, eval_size


def train_model(model, optimizer, train_loader, val_loader, device):
    model.train()
    experiment_dir = f'experiments/{EXPERIMENT_NAME}'
    os.makedirs(experiment_dir, exist_ok=True)

    # Add this line to keep track of the lowest validation loss
    lowest_val_loss = float('inf')

    with open(os.path.join(experiment_dir, 'training_logs.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Add Timestamp and Batch Size in the header
        csv_writer.writerow(
            ['Iteration', 'Phase', 'Mean Loss', 'Speed Error', 'Gear Accuracy', 'Timestamp', 'Batch Size'])

        progress_bar = tqdm(total=NUM_ITERS, desc='Training')

        mean_loss = 0.0
        iter_count = 0
        mean_speed_loss = 0.0
        mean_gear_loss = 0.0
        mean_speed_error = 0.0
        mean_gear_accuracy = 0.0

        while iter_count < NUM_ITERS:
            for i, batch in enumerate(train_loader):
                images, in_race, true_speed, true_position, true_lap, true_gear = batch
                images, in_race, true_speed, true_position, true_lap, true_gear = \
                    images.to(device), in_race.to(device), true_speed.to(device), \
                        true_position.to(device), true_lap.to(device), true_gear.to(device)

                optimizer.zero_grad()

                pred_in_race, pred_speed, pred_position, pred_gear, pred_lap = model(images)

                # Calculate loss
                speed_loss = 5 * nn.MSELoss()(pred_speed.squeeze(), true_speed.float())
                gear_loss = nn.CrossEntropyLoss()(pred_gear, true_gear.long())
                total_loss = speed_loss + gear_loss
                speed_error = calculate_speed_error(pred_speed, true_speed).item()
                gear_accuracy = calculate_accuracy(pred_gear, true_gear)

                total_loss.backward()
                optimizer.step()

                # Update the exponential moving averages
                mean_loss = total_loss.item() if iter_count == 0 else (1 - DECAY) * total_loss.item() + DECAY * mean_loss
                mean_speed_loss = speed_loss if iter_count == 0 else (1 - DECAY) * speed_loss + DECAY * mean_speed_loss
                mean_gear_loss = gear_loss if iter_count == 0 else (1 - DECAY) * gear_loss + DECAY * mean_gear_loss
                mean_speed_error = speed_error if iter_count == 0 else (1 - DECAY) * speed_error + DECAY * mean_speed_error
                mean_gear_accuracy = gear_accuracy if iter_count == 0 else (1 - DECAY) * gear_accuracy + DECAY * mean_gear_accuracy

                progress_bar.set_postfix({
                    'loss': f'{mean_loss:.3f}',
                    'loss_speed': f'{mean_speed_loss:.3f}',
                    'loss_gear': f'{mean_gear_loss:.3f}',
                    'speed_err': f'{mean_speed_error:.2f}mph',
                    'gear_acc': f'{mean_gear_accuracy:.2%}',
                    # 'eval_time': f'{eval_duration:.2f}s',
                    # 'eval_size': f'{eval_size:.0f}'
                })
                progress_bar.update()

                # Write training metrics to CSV file with timestamp and batch size
                csv_writer.writerow([iter_count, 'training', total_loss.item(), speed_error, gear_accuracy,
                                     time.strftime('%Y-%m-%d %H:%M:%S'), images.shape[0]])
                iter_count += 1

                if iter_count % EVAL_ITERS == 0:
                    # Evaluate on the validation set
                    avg_val_loss, avg_val_speed_error, avg_val_gear_accuracy, eval_duration, eval_size = evaluate_model(
                        model, val_loader, device)

                    # Write validation metrics to CSV file with timestamp and batch size
                    csv_writer.writerow(
                        [iter_count, 'validation', avg_val_loss.item(), avg_val_speed_error, avg_val_gear_accuracy,
                         time.strftime('%Y-%m-%d %H:%M:%S'), eval_size])

                    # Save the model if the validation loss improves
                    if avg_val_loss.item() < lowest_val_loss:
                        lowest_val_loss = avg_val_loss.item()
                        torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))

                if iter_count >= NUM_ITERS:
                    break

        # Save the last model
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'last_model.pth'))
        progress_bar.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_loaders(CSV_FILE_PATH, FRAMES_DIR, BATCH_SIZE)
    train_loader = loaders.get('train')
    val_loader = loaders.get('valid')

    model = RacingNet(freeze_backbone=FREEZE_BACKBONE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_model(model, optimizer, train_loader, val_loader, device)


if __name__ == "__main__":
    main()

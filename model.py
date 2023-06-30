import torch
import torch.nn as nn
import torchvision.models as models


class RacingNet(nn.Module):
    def __init__(self, num_positions=20, num_gears=10, num_laps=100, freeze_backbone=True, use_pretrained=True):
        super(RacingNet, self).__init__()

        # Load pre-trained MobileNetV3-Small model
        model_weights = 'IMAGENET1K_V1' if use_pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=model_weights)

        # Freeze the layers of the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the last fully connected layer to match the desired outputs
        num_features = 1000

        # Output for in_race (binary)
        self.in_race_fc = nn.Linear(num_features, 1)

        # Output for speed (continuous)
        self.speed_fc = nn.Linear(num_features, 1)

        # Output for position (categorical)
        self.position_fc = nn.Linear(num_features, num_positions)

        # Output for gear (categorical)
        self.gear_fc = nn.Linear(num_features, num_gears)

        # Output for lap (categorical)
        self.lap_fc = nn.Linear(num_features, num_laps)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Forward pass for each output
        in_race = torch.sigmoid(self.in_race_fc(features))
        speed = torch.sigmoid(self.speed_fc(features))
        position = torch.softmax(self.position_fc(features), dim=1)
        gear = torch.softmax(self.gear_fc(features), dim=1)
        lap = torch.softmax(self.lap_fc(features), dim=1)

        return in_race, speed, position, gear, lap


def main():
    net = RacingNet()

    # Count the total number of parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # Print the number of parameters
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Pass example input through the model
    example_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 images
    in_race, speed, position, gear, lap = net(example_input)

    # Print the shapes of the output tensors
    print(f"in_race output shape: {in_race.shape}")
    print(f"speed output shape: {speed.shape}")
    print(f"position output shape: {position.shape}")
    print(f"gear output shape: {gear.shape}")
    print(f"lap output shape: {lap.shape}")


if __name__ == "__main__":
    main()

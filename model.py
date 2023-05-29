import torch
import torch.nn as nn
import torchvision.models as models


class RacingNet(nn.Module):
    def __init__(self, num_positions=20, num_gears=10, num_laps=100):
        super(RacingNet, self).__init__()

        # Load pre-trained ResNet-34 model
        self.backbone = models.resnet34(pretrained=True)

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

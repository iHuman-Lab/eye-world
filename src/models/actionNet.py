import torch
import torch.nn as nn


class ActionNet(nn.Module):
    def __init__(self, num_actions):
        super(ActionNet, self).__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Atari style input
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Fully connected action head
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        x = self.fc_layers(x)

        return x  # logits

    import pytorch_lightning as pl

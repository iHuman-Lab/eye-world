import torch
from torch import nn

# ================================================================
# Action network
# ================================================================


class ActionNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        # An 84x84 input produces 64 x 7 x 7 = 3136 features.
        self.num_actions = num_actions
        self.num_predictions = 1

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                self.num_predictions * self.num_actions,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        # [batch, 36] -> [batch, 4, 9]
        x = x.reshape(
            x.shape[0],
            self.num_predictions,
            self.num_actions,
        )

        return x

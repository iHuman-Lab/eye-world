import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        in_channels = config.get("stack_length", 1)
        self.conv1 = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 9, kernel_size=3, padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(9, 18, kernel_size=3, padding=1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(18, 22, kernel_size=3, padding=1),
            nn.BatchNorm2d(22),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(22, 26, kernel_size=3, padding=1),
            nn.BatchNorm2d(26),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(26, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            # Block 6 (final)
            nn.Conv2d(30, 34, kernel_size=3, padding=1),
            nn.BatchNorm2d(34),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
        )

        with torch.no_grad():
            if config.get("grey_scale", True):
                x = torch.ones(1, 1, config["size_x"], config["size_y"])
            else:
                x = torch.ones(1, 3, config["size_x"], config["size_y"])

            out = self.conv1(x)
            features = out.numel()

        self.lin1 = nn.Sequential(
            nn.Linear(features, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 2),
        )

    def forward(self, img):
        output = self.conv1(img)
        output = output.view(output.size(0), -1)

        output = self.lin1(output)
        return output


# TODO: Add a convolution and de-convolution network architecture
class DeConvNet(nn.Module):
    def __init__(self, config):
        super(DeConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            # Block 1
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # Block 4
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # Block 5
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Block 6 (final)
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=4),
            nn.Sigmoid(),
        )

        with torch.no_grad():
            self.eval()
            if config.get("grey_scale", True):
                x = torch.ones(1, 4, config["size_x"], config["size_y"])
            else:
                x = torch.ones(1, 12, config["size_x"], config["size_y"])

            out = self.conv1(x)
            features = out.view(1, -1).size(1)

        self.lin1 = nn.Sequential(
            nn.Linear(features, 500),
            nn.Linear(500, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 2),
        )

    def forward(self, img):
        output = self.conv1(img)
        output = output.view(output.size(0), -1)

        output = self.lin1(output)
        return output

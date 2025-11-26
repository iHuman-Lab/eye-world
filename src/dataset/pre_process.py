from collections import deque

import torch
from torchvision import transforms

from .eye_gaze_process import eye_gaze_to_density_image


class Resize:
    def __init__(self, config):
        if config.get("grey_scale", True):
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                    ),
                ]
            )

    def __call__(self, sample):
        img, eye_gazes = sample
        img = self.transform(img)

        return img, eye_gazes


class Stack:
    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.stack = deque(maxlen=self.stack_len)
        self.config = config

    def __call__(self, sample):
        img, eye_gazes = sample
        if len(self.stack) < self.stack_len:
            while len(self.stack) < self.stack_len:
                self.stack.append(img)
        else:
            self.stack.append(img)
        stacked = torch.cat(list(self.stack), dim=0)
        density_image = eye_gaze_to_density_image(img.shape, eye_gazes, self.config)

        return stacked, density_image


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)
        return sample

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
        img, eye_gazes, action = sample
        img = self.transform(img)

        return img, eye_gazes, action


class Stack:
    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.stack = deque(maxlen=self.stack_len)
        self.config = config

    def __call__(self, sample):
        img, eye_gazes, action = sample
        if len(self.stack) < self.stack_len:
            while len(self.stack) < self.stack_len:
                self.stack.append(img)
        else:
            self.stack.append(img)
        stacked = torch.cat(list(self.stack), dim=0)
        if self.config.get("eye_density", False):
            gaze_out = eye_gaze_to_density_image(img.shape, eye_gazes, self.config)
        else:
            gaze_out = eye_gazes[-1]

        return stacked, gaze_out, action


class StackWithLabels:
    """
    Stacks images, gaze, and actions in temporal order. Keeps everything aligned with stack_len frames.
    """

    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.config = config
        self.img_stack = deque(maxlen=self.stack_len)
        self.gaze_stack = deque(maxlen=self.stack_len)
        self.action_stack = deque(maxlen=self.stack_len)

    def __call__(self, sample):
        img, eye_gaze, action = sample

        if len(self.img_stack) < self.stack_len:
            while len(self.img_stack) < self.stack_len:
                # Append current sample
                self.img_stack.append(img)
                self.gaze_stack.append(eye_gaze)
                self.action_stack.append(action)
        else:
            self.img_stack.append(img)
            self.gaze_stack.appendleft(eye_gaze)
            self.action_stack.appendleft(action)

        # Stack images along channel dimension: (C, H, W) -> (stack_len*C, H, W)
        stacked_img = torch.cat(list(self.img_stack), dim=0)

        # Stack actions as tensor (stack_len,)
        stacked_action = torch.tensor(list(self.action_stack), dtype=torch.long)

        # Stack gaze
        if self.config.get("eye_density", False):
            # For density maps, generate one per frame and stack channels
            gaze_maps = [
                eye_gaze_to_density_image(img.shape, gaze, self.config)
                for gaze in self.gaze_stack
            ]
            stacked_gaze = torch.cat(gaze_maps, dim=0)
        else:
            # For raw gaze points: (stack_len, 2)
            stacked_gaze = torch.tensor(self.gaze_stack, dtype=torch.float32)

        return stacked_img, stacked_gaze, stacked_action


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)

        return sample

from collections import deque

import torch
from torchvision import transforms

"""class Resize:
    def __init__(self, config):
        self.grey_scale = config.get("grey_scale_v", True)
        self.size = (config["size_x"], config["size_y"])  # (H, W)

    def __call__(self, sample):
        img, eye_gazes = sample  # img: [C,H,W] or [B,C,H,W]

        # Ensure batch dim
        if img.ndim == 3:  # [C,H,W]
            img = img.unsqueeze(0)  # [1,C,H,W]

        # Resize
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        # Convert to grayscale with channel retained
        if self.grey_scale:
            if img.shape[1] == 3:
                # weighted sum, keep channel dim
                img = (
                    0.2989 * img[:, 0:1, :, :]
                    + 0.5870 * img[:, 1:2, :, :]
                    + 0.1140 * img[:, 2:3, :, :]
                )
            # img now shape [B,1,H,W]

        # Normalize 0-1
        img_min = img.amin(dim=(1, 2, 3), keepdim=True)
        img_max = img.amax(dim=(1, 2, 3), keepdim=True)
        img = (img - img_min) / (img_max - img_min + 1e-8)

        # Remove batch dim if input was single image
        if img.shape[0] == 1:
            img = img.squeeze(0)  # [1,H,W] still keeps channel dim

        # Ensure channel dimension exists
        if img.ndim == 2:  # [H,W] -> [1,H,W]
            img = img.unsqueeze(0)

        return img, eye_gazes
"""

"""
class Resize:
    def __init__(self, config):
        self.grey_scale = config.get("Grey_scale_V", True)
        self.size = (config["size_x"], config["size_y"])  # (H, W)

    def __call__(self, sample):
        img, eye_gazes = sample  # img: [B, C, H, W] or [C, H, W]

        # If single image without batch dim, add batch dim
        if img.ndim == 3:  # [C, H, W]
            img = img.unsqueeze(0)  # [1, C, H, W]

        # Resize using F.interpolate
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        # Convert to grayscale if requested
        if self.grey_scale:
            if img.shape[1] == 3:
                # RGB -> grayscale: 0.2989 R + 0.5870 G + 0.1140 B
                img = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
            # else assume already 1 channel

        # Normalize to 0-1
        img_min = img.amin(dim=(1, 2, 3), keepdim=True)
        img_max = img.amax(dim=(1, 2, 3), keepdim=True)
        img = (img - img_min) / (img_max - img_min + 1e-8)

        # Remove batch dim if single image
        if img.shape[0] == 1:
            img = img.squeeze(0)  # [1, H, W] or [C, H, W]

        return img, eye_gazes
        """


class Resize:
    def __init__(self, config):
        if config.get("Grey_scale_v", True):
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


"""class Stack:
    def __init__(self, config):
        self.stack_len = config["stack_size"]
        self.stack = deque(maxlen=self.stack_len)
        self.config = config

    def __call__(self, sample):
        img, eye_gazes = sample

        # Append the new image to the deque
        self.stack.append(img)

        # If deque is not full yet, repeat the first frame to fill it
        if len(self.stack) < self.stack_len:
            # repeat first image to fill stack
            repeat_count = self.stack_len - len(self.stack)
            stacked = torch.cat(list(self.stack) + [img] * repeat_count, dim=0)
        else:
            stacked = torch.cat(list(self.stack), dim=0)

        return stacked, eye_gazes
"""


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

        return stacked, eye_gazes[-1]


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)

        return sample

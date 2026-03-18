from collections import deque

import torch

from dataset.eye_gaze_process import (
    eye_gaze_to_density_image,  # needed for gaze heatmaps
)

print("a")
# src/dataset/pre_process_jepa.py


# -----------------------------
# 1. Stack Preprocessor
# -----------------------------
class StackWithLabels:
    """
    Stacks images, gaze, and actions in temporal order.
    Keeps everything aligned with stack_len frames.
    """

    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.config = config
        self.img_stack = deque(maxlen=self.stack_len)
        self.gaze_stack = deque(maxlen=self.stack_len)
        self.action_stack = deque(maxlen=self.stack_len)

    def __call__(self, sample):
        img, eye_gaze, action = sample

        # Append current sample
        self.img_stack.append(img)
        self.gaze_stack.append(eye_gaze)
        self.action_stack.append(action)

        # Pad beginning if stack not full
        while len(self.img_stack) < self.stack_len:
            self.img_stack.appendleft(img)
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
                eye_gaze_to_density_image(img.shape, g, self.config)
                for img, g in zip(self.img_stack, self.gaze_stack)
            ]
            stacked_gaze = torch.cat(gaze_maps, dim=0)
        else:
            # For raw gaze points: (stack_len, 2)
            stacked_gaze = torch.tensor(self.gaze_stack, dtype=torch.float32)

        return stacked_img, stacked_gaze, stacked_action


# -----------------------------
# 2. Compose Preprocessor
# -----------------------------
class ComposePreprocessor:
    """
    Chains multiple preprocessors sequentially.
    Example:
        preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
    """

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)
        return sample

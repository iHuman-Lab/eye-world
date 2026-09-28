import random
import sys
from contextlib import contextmanager

import torch
from tensorboard import program
from torch.utils.data import IterableDataset

# from utils import InterleavedDataset, skip_run

# The configuration file


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        activated = ["run"]
        p = ColorPrint()  # printing options
        if flag in activated:
            p.print_run("{:>12}  {:>3}  {:>12}".format("Running the block", "|", f))
            yield
        else:
            p.print_skip("{:>12}  {:>2}  {:>12}".format("Skipping the block", "|", f))
            raise SkipWith()

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end="\n"):
        sys.stderr.write("\x1b[88m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_run(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_warn(message, end="\n"):
        sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)


def get_num_gpus():
    if torch.cuda.device_count() == 0:
        return None
    else:
        return list(range(torch.cuda.device_count()))


def launch_tensorboard(hparams):
    tb = program.TensorBoard()
    tb.configure(
        argv=[
            None,
            "--logdir",
            hparams.log_dir,
            "--reload_multifile",
            "true",
            "--reload_interval",
            "15",
        ]
    )
    tb.launch()

    # --------------------------------------------------
    # Interleave samples from multiple games
    # --------------------------------------------------


class InterleavedDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]

        while iterators:
            idx = random.randrange(len(iterators))

            try:
                yield next(iterators[idx])

            except StopIteration:
                iterators.pop(idx)

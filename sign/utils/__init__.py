import random

import dgl
import numpy as np
import torch

from .data import *
from .metapath import *
from .metrics import *
from .neg_sampler import *
from .random_walk import *


def set_random_seed(seed):
    """Set random seed for Python, numpy, PyTorch

    :param seed: int Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)


def get_device(device):
    """Return the specified GPU device

    :param device: int GPU index, -1 means CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')

from abc import ABCMeta, abstractmethod
import os
from torch.autograd import Variable
import torch
from torchvision.datasets import CIFAR100

from .build import CLIP_REGISTRY
from xmodaler.config import kfg
from xmodaler.config import CfgNode as CN

__all__ = ["ClipFilter"]

@CLIP_REGISTRY.register()
class ClipFilter(nn.model, metaclass=ABCMeta):
    @configurable
    def __init__(
            self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    def preprocess(self, batched_inputs):
        return batched_inputs







# Load the model

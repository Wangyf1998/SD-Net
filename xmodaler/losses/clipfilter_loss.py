import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.config import CfgNode as CN
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class ClipFilter(nn.Module):
    @configurable
    def __init__(self, filter_weight, reconstruct_weight, slot_size, num_classes):
        super(ClipFilter, self).__init__()
        # weight = torch.ones((num_classes + 1,)).cuda()
        # weight[-1] = 30.0
        self.criterion = nn.CrossEntropyLoss()
    @classmethod
    def from_config(cls, cfg):
        return {

        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret = {}
        logits = outputs_dict[kfg.CONCEPTS_PRED]
        concepts_labels = outputs_dict[kfg.CONCEPTS_LABELS]
        filter_loss = self.criterion(logits, concepts_labels)
        ret.update({
            "filter_loss": filter_loss})




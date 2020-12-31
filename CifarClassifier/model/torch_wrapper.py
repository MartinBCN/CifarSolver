from pathlib import Path
from typing import Union
import torch
from torch import nn
import torch.optim as optim


class TorchWrapper:
    def __init__(self, name: str):
        self.name = name
        if self.model is None:
            self.model = nn.Module()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

        self.training_log = {}

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    @classmethod
    def load(cls, filepath: Union[str, Path], name: str = None):
        state = torch.load(filepath)

        new = cls(name)

        new.set_optimizer()

        new.model.load_state_dict(state['model_state_dict'])
        new.optimizer.load_state_dict(state['optimizer_state_dict'])
        new.training_log = state['training_log']

        return new

    def save(self, filepath: Union[str, Path]):

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'training_log': self.training_log
                 }

        torch.save(state, filepath)


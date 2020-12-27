from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from torch import nn

from CifarClassifier.model.cnn import CifarCNN
import torch.optim as optim


class TorchWrapper:
    def __init__(self):
        self.model = CifarCNN()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.training_log = {}

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        state = torch.load(filepath)

        new = cls()

        new.model.load_state_dict(state['state_dict'])
        new.optimizer.load_state_dict(state['optimizer'])
        new.training_log = state['training_log']

        return new

    def save(self, filepath: Union[str, Path]):

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'training_log': self.training_log
                 }

        torch.save(state, filepath)

    def train(self, dataloader: dict, epochs: int, early_stop_epochs: int = 5):

        start = datetime.now()
        for epoch in range(epochs):
            for phase in ['train', 'validation']:
                if phase not in self.training_log.keys():
                    self.training_log[phase] = {'epoch_loss': [], 'batch_loss': []}
                epoch_loss = 0
                for i, (image, label) in enumerate(dataloader[phase]):
                    if phase == 'train':
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image)
                    loss = self.criterion(outputs, label)
                    epoch_loss += loss

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    if 'batch_loss' in self.training_log[phase].keys():
                        self.training_log[phase]['batch_loss'].append(loss)
                    else:
                        self.training_log[phase]['batch_loss'] = [loss]

                    if (i % 50) == 0:
                        print(f'Batch {i}/{len(dataloader[phase])}, time elapsed: {datetime.now() - start}')

                if 'epoch_loss' in self.training_log[phase].keys():
                    self.training_log[phase]['epoch_loss'].append(epoch_loss)
                else:
                    self.training_log[phase]['epoch_loss'] = [epoch_loss]

from model.cnn import CifarCNN
from model.torch_wrapper import TorchWrapper
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from data.transformer import image_to_tensor
from data.data_loader import classes


class ImageClassifier(TorchWrapper):
    classes = classes

    def __init__(self, name: str):
        if name == 'cnn':
            self.model = CifarCNN()
        super().__init__(name)

    def evaluate(self, data_loader: DataLoader):

        predicted_labels = []
        ground_truth = []

        for i, (image, label) in enumerate(data_loader):

            with torch.no_grad():

                predicted_labels.append(self.model.predict(image).detach().cpu().numpy())
                ground_truth.append(label.detach().cpu().numpy())

        # Log accuracy
        predicted_labels = np.concatenate(predicted_labels)
        ground_truth = np.concatenate(ground_truth)
        accuracy = (ground_truth == predicted_labels).mean()
        return accuracy, predicted_labels, ground_truth

    def train(self, dataloader: dict, epochs: int, early_stop_epochs: int = 5):

        start = datetime.now()
        for epoch in range(epochs):
            for phase in ['train', 'validation']:
                if phase not in self.training_log.keys():
                    self.training_log[phase] = {'epoch_loss': [], 'batch_loss': []}

                epoch_loss = 0
                epoch_ground_truth = []
                epoch_predicted_labels = []

                for i, (image, label) in enumerate(dataloader[phase]):
                    if phase == 'train':
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image)
                    loss = self.criterion(outputs, label)
                    epoch_loss += loss.detach().cpu().numpy()

                    epoch_predicted_labels.append(self.model.predict(image).detach().cpu().numpy())
                    epoch_ground_truth.append(label.detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    if 'batch_loss' in self.training_log[phase].keys():
                        self.training_log[phase]['batch_loss'].append(loss.detach().cpu().numpy())
                    else:
                        self.training_log[phase]['batch_loss'] = [loss.detach().cpu().numpy()]

                # Log accuracy
                accuracy = (np.concatenate(epoch_ground_truth) == np.concatenate(epoch_predicted_labels)).mean()
                if 'accuracy' in self.training_log[phase].keys():
                    self.training_log[phase]['accuracy'].append(accuracy)
                else:
                    self.training_log[phase]['accuracy'] = [accuracy]

                # Log Epoch Loss
                if 'epoch_loss' in self.training_log[phase].keys():
                    self.training_log[phase]['epoch_loss'].append(epoch_loss)
                else:
                    self.training_log[phase]['epoch_loss'] = [epoch_loss]

    def predict(self, image: Image):
        t = image_to_tensor(image)
        t = t.unsqueeze(0)
        label = self.model.predict(t).detach().cpu().numpy()
        return int(label)

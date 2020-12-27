from pathlib import Path

from data.data_loader import get_data_loader
from model.torch_wrapper import TorchWrapper
import matplotlib.pyplot as plt

model = TorchWrapper()

data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

train_loader = get_data_loader('train', data)

model.train(train_loader, 2)

plt.subplot(2, 2, 1)
plt.plot(model.training_log['train']['batch_loss'])

plt.subplot(2, 2, 2)
plt.plot(model.training_log['train']['epoch_loss'])

plt.subplot(2, 2, 3)
plt.plot(model.training_log['validation']['batch_loss'])

plt.subplot(2, 2, 4)
plt.plot(model.training_log['validation']['epoch_loss'])

plt.show()
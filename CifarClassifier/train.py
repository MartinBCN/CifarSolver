import itertools
from pathlib import Path

from data.data_loader import get_data_loader
from model.image_classifier import ImageClassifier
import matplotlib.pyplot as plt

model_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/models'
figure_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/figures'
data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

name = 'test1'

model = ImageClassifier('cnn')
model.set_optimizer()

train_loader = get_data_loader('train', data)

model.train(train_loader, 10)

phases = ['train', 'validation']
fields = ['batch_loss', 'epoch_loss', 'accuracy']


for i, (phase, field) in enumerate(itertools.product(*[phases, fields]), start=1):

    plt.subplot(2, 3, i)
    plt.plot(model.training_log[phase][field])

plt.savefig(f'{figure_dir}/{name}.png')

model.save(f'{model_dir}/{name}.ckpt')

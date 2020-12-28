from pathlib import Path

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

from data.data_loader import get_data_loader
from model.torch_wrapper import TorchWrapper
import matplotlib.pyplot as plt
from data.data_loader import classes


model_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/models'
figure_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/figures'
data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

name = 'test1'


model = TorchWrapper.load(f'{model_dir}/{name}.ckpt')

test = get_data_loader(phase='test', data_dir=data)

test = test['test']

accuracy, predicted_labels, ground_truth = model.evaluate(test)

cm = confusion_matrix([classes[i] for i in ground_truth],
                      [classes[i] for i in predicted_labels], normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(include_values=True, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()


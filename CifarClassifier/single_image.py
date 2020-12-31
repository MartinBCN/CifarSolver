from pathlib import Path
import random
from PIL import Image

from model.image_classifier import ImageClassifier


model_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/models'
figure_dir = '/home/martin/Programming/Python/DeepLearning/Cifar10/figures'
data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

name = 'test1'

files = data.rglob('*.png')
files = [x for x in files if x.is_file()]
image_classifier = ImageClassifier.load(filepath=f'{model_dir}/{name}.ckpt', name='cnn')

# use local version of classifier
for i in range(10):
    fn = random.choice(files)
    label = fn.parents[0].name
    image = Image.open(fn)
    prediction = image_classifier.predict(image)
    print(label, prediction)




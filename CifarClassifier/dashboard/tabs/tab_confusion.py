from sklearn.metrics import confusion_matrix
import json
import os
from pathlib import Path
import requests
import dash_html_components as html

data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw/test')
static_image_route = '/static/'

file_names = data.rglob('*.png')
file_names = [x for x in file_names if x.is_file()]


def build_confusion():

    true_labels = []
    predicted_labels = []

    for (i, fn) in enumerate(file_names):

        if (i % 10) == 0:
            print(f'{i} / {len(file_names)}')

        true_labels.append(fn.parents[0].name.lower())

        # Send image
        url = 'http://localhost:8000/predict'

        name_img = os.path.basename(fn)

        files = {
            'file': (name_img,
                     open(fn, 'rb').read(),
                     "image/png")
        }

        with requests.Session() as s:
            r = s.post(url, files=files)
            prediction = json.loads(r.content)['likely_class']

            predicted_labels.append(prediction)

    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')

    print(cm)

    return html.Div(children='test')

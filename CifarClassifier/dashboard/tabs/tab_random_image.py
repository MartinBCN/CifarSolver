import base64
import json
import os
from pathlib import Path
import random

import flask
from PIL import Image
import requests
from dash.dependencies import Output, Input

from dashboard.maindash import app
import dash_html_components as html

data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')
static_image_route = '/static/'

file_names = data.rglob('*.png')
file_names = [x for x in file_names if x.is_file()]


def build_random_image():

    fn = random.choice(file_names)

    image_filename = f'{fn}'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

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
        likely_class = json.loads(r.content)['likely_class']

    img_div = html.Img(
        src='data:image/png;base64,{}'.format(encoded_image.decode()),
        id="random-image",
        style={
            "height": "60px",
            "width": "auto",
            "margin-bottom": "25px",
        },
    )

    return html.Div(children=[img_div, html.A(likely_class)])

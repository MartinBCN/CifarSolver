import json

import curlify
import os
from pathlib import Path
import random
from PIL import Image
import requests

data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

name = 'test1'

files = data.rglob('*.png')
files = [x for x in files if x.is_file()]

# Two: use FastAPI
fn = random.choice(files)
label = fn.parents[0].name
image = Image.open(fn)

# Test if server is up
# I have added the hello world to the root for this purpose
r = requests.get('http://localhost:8000/')
print(r.status_code)
print(r.text)

# Send image
url = 'http://localhost:8000/predict'


name_img = os.path.basename(fn)

files = {
    'file': (name_img,
             open(fn,'rb').read(),
             "image/png")
}

print(fn)

with requests.Session() as s:
    r = s.post(url, files=files)
    print(r.status_code)

    print(r.text)
    print(json.loads(r.content)['likely_class'])

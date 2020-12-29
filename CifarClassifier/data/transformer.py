from PIL import Image

import torchvision.transforms.functional as tf


def image_to_tensor(image: Image):
    image = tf.to_tensor(image)
    image = tf.normalize(image, (0.5,), (0.5,))

    return image

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as tf


classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

CLASSES = {l: i for (i, l) in enumerate(classes)}


class CifarDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path]):
        if type(data_dir) is str:
            data_dir = Path(data_dir)

        self.files = []
        self.labels = []
        for file in data_dir.rglob('*.png'):
            self.files.append(file)
            self.labels.append(file.parents[0].name)

    def transform(self, image: Image, label: str):

        image = tf.to_tensor(image)
        image = tf.normalize(image, (0.5,), (0.5, ))
        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        label = torch.tensor(CLASSES[label.lower()])

        return image, label

    def __getitem__(self, item):
        image = self.files[item]
        image = Image.open(image).convert('RGB')
        label = self.labels[item]

        image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.labels)


def get_data_loader(phase: str, data_dir: Union[str, Path], batch_size: int = 4) -> dict:
    dataset = CifarDataset(data_dir / phase.lower())
    if phase == 'test':
        data_loader = {'test': DataLoader(dataset, batch_size=batch_size)}
    else:
        split = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
        train, validation = torch.utils.data.random_split(dataset, split, generator=torch.Generator().manual_seed(42))
        data_loader = {'train': DataLoader(train, batch_size=batch_size),
                       'validation': DataLoader(validation, batch_size=batch_size)}
    return data_loader


if __name__ == '__main__':
    data = Path('/home/martin/Programming/Python/DeepLearning/Cifar10/data/cifar10_raw')

    loader = get_data_loader('train', data)

    print(len(loader['train']))
    print(len(loader['validation']))
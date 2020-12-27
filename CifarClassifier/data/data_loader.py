import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class CifarLoader(object):

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_train_validation(self):

        trainset = CIFAR10(root=os.environ['DATA_DIR'], train=True, download=True, transform=self.transform)
        print(type(trainset))

        trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        return trainloader

    def get_test(self):

        testset = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        return testloader


if __name__ == '__main__':
    os.environ['DATA_DIR'] = '/home/martin/Programming/Python/DeepLearning/Cifar10/data'
    loader = CifarLoader()
    train = loader.get_train_validation()

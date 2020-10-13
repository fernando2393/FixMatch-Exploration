import json
import collections
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks
def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())

    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_loaders(dataset, train_batch_size=128, test_batch_size=10):

    if dataset == 'cifar10':

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    elif dataset == 'svhn':

        normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    return trainloader, testloader


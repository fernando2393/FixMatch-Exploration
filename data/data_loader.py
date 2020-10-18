# Data manipulation
import numpy as np
from PIL import Image

# Access CIFAR-10, MNIST and SVHN
import torch
from torchvision import datasets
from torchvision import transforms
from CTAugment import augment

# Pre-defining mean and std for the datasets to reduce computational time
CIFAR10_mean = (0.4914, 0.4822, 0.4465)
CIFAR10_std = (0.2471, 0.2435, 0.2616)
MNIST_mean = 0.1307
MNIST_std = 0.3081
SVHN_mean = (0.4377, 0.4438, 0.4728)
SVHN_std = (0.1980, 0.2010, 0.1970)

# Set randomness reproducibility
np.random.seed(42)


# -----TRANSFORMATIONS----- #
def tensor_normalizer(mean, std):
    # Normalizing the testing images
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def weakly_augmentation(mean, std):
    # Perform weak transformation on labeled and unlabeled training images
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return weak_transform


def cta_augmentation_labeled_data(mean, std, cta):
    # Perform weak transformation on labeled and unlabeled training images

    cta_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        augment(cta, True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return cta_transform


def split_labeled_unlabeled(num_labeled, n_classes, labels, balanced_split=True):
    labeled_indeces = []
    unlabeled_indeces = []
    if balanced_split:
        # Define number of labeled samples per class
        lsamples_per_class = num_labeled // n_classes
        # Get indeces of each class to make it balanced based on lsamples_per_class
        for i in range(n_classes):
            tmp_indeces = np.where(labels == i)[0]
            np.random.shuffle(tmp_indeces)
            labeled_indeces.extend(tmp_indeces[:lsamples_per_class])
            unlabeled_indeces.extend(tmp_indeces[lsamples_per_class:])

    else:
        print("TO DO: DEFINE UNBALANCED DATA SETS")
        exit(0)

    return labeled_indeces, unlabeled_indeces


def applyTransformations(root, labeled_indeces_extension, labeled_indeces ,unlabeled_indeces, mean, std, cta):
    # Transform label data -> weak transformation
    train_labeled_data = DataTransformation(root, labeled_indeces_extension, train=True, transform=weakly_augmentation(mean, std))

    # Transform label data -> strong transformation (CTA)
    train_labeled_data_cta = DataTransformation(root, labeled_indeces, train=True,
                                                transform=cta_augmentation_labeled_data(mean, std, cta))

    # Transform unlabeled data -> weak transformationa and CTAugment
    train_unlabeled_data = DataTransformation(root, unlabeled_indeces, train=True,
                                              transform=SSLTransform(mean, std, cta))

    return train_labeled_data, train_unlabeled_data, train_labeled_data_cta


# -----CONSTRUCT DATA OBJECTS----- #
class DataTransformation(datasets.CIFAR10):
    def __init__(self, root, indeces, train=True, transform=None, target_transform=None, download=False):
        # Accessing CIFAR10 from torchvision
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indeces is not None:
            self.data = self.data[indeces]
            self.targets = np.array(self.targets)[indeces]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# -----UNLABELED DATA WEAKLY & STRONGLY AUGMENTATION----- #
class SSLTransform(object):
    def __init__(self, mean, std, cta):
        # Weakly Data Augmentation
        self.weakly = weakly_augmentation(mean, std)
        # Strongly Data Augmentation
        self.strongly = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            augment(cta, False),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weakly_augment = self.weakly(x)
        strongly_augment = self.strongly(x)
        return weakly_augment, strongly_augment


# -----LOADING DATA----- #
# Load CIFAR-10
def load_cifar10(root, num_labeled):
    # Import data and define labels
    raw_data = datasets.CIFAR10(root, train=True, download=True)
    labels = np.array(raw_data.targets)

    # Import test data
    test_data = datasets.CIFAR10(root, train=False, transform=tensor_normalizer(mean=CIFAR10_mean, std=CIFAR10_std),
                                 download=False)

    # split data into labeled and unlabeled
    labeled_indeces, unlabeled_indeces = split_labeled_unlabeled(
        num_labeled,
        n_classes=10,
        labels=labels,
        balanced_split=True
    )

    return labeled_indeces, unlabeled_indeces, test_data

# Load SVHN

# Load MNIST

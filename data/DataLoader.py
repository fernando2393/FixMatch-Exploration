# Data manipulation
import numpy as np
import Constants as cts
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from CTAugment import augment, cutout_strong

# Set randomness reproducibility
np.random.seed(42)


# -----TRANSFORMATIONS----- #
def tensor_normalizer(mean, std):
    # Normalizing the testing images
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def weakly_augmentation(mean, std):
    # Perform weak transformation on labeled and unlabeled training images
    if cts.DATASET[0] != "SVHN":
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        weak_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return weak_transform


def cta_augmentation_labeled_data(mean, std, cta):
    # Perform weak transformation on labeled and unlabeled training images

    cta_transform = transforms.Compose([
        augment(cta, True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return cta_transform


def split_labeled_unlabeled(num_labeled, n_classes, labels, balanced_split=True, unbalance=0, unbalanced_proportion=1.0,
                            sample_proportion=1):
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
            unlabeled_indeces.extend(tmp_indeces[lsamples_per_class:int(len(tmp_indeces) * sample_proportion)])

    else:
        lsamples_per_class = num_labeled // n_classes
        unbalanced_class = int(lsamples_per_class * unbalanced_proportion)
        for i in range(n_classes):
            tmp_indeces = np.where(labels == i)[0]
            np.random.shuffle(tmp_indeces)
            if i == unbalance:
                labeled_indeces.extend(tmp_indeces[:unbalanced_class])
                if unbalanced_proportion < 1.0:
                    unlabeled_indeces.extend(tmp_indeces[lsamples_per_class:int(len(tmp_indeces) * sample_proportion)])
                else:
                    unlabeled_indeces.extend(tmp_indeces[unbalanced_class:int(len(tmp_indeces) * sample_proportion)])
            else:
                labeled_indeces.extend(tmp_indeces[:lsamples_per_class])
                unlabeled_indeces.extend(tmp_indeces[lsamples_per_class:int(len(tmp_indeces) * sample_proportion)])

    return labeled_indeces, unlabeled_indeces


def applyTransformations(root, labeled_indeces_extension, labeled_indeces, unlabeled_indeces, mean, std, cta):
    # Transform label data -> weak transformation
    train_labeled_data = DataTransformation(root, labeled_indeces_extension,
                                            transform=weakly_augmentation(mean, std))

    # Transform label data -> strong transformation (CTA)
    train_labeled_data_cta = DataTransformation(root, labeled_indeces,
                                                transform=cta_augmentation_labeled_data(mean, std, cta))

    # Transform unlabeled data -> weak transformationa and CTAugment
    train_unlabeled_data = DataTransformation(root, unlabeled_indeces,
                                              transform=SSLTransform(mean, std, cta))

    return train_labeled_data, train_unlabeled_data, train_labeled_data_cta


# -----CONSTRUCT DATA OBJECTS----- #
class DataTransformation(cts.DATASET[3]):
    def __init__(self, root, indeces, transform=None, target_transform=None, download=True):
        # Accessing CIFAR10 from torchvision
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        if indeces is not None:
            self.data = self.data[indeces]
            if hasattr(self, 'targets'):
                self.targets = np.array(self.targets)[indeces]
            else:
                self.labels = np.array(self.labels)[indeces]

    def __getitem__(self, index):
        if hasattr(self, 'targets'):
            img, target = self.data[index], self.targets[index]
            if cts.DATASET[0] == "MNIST":
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        else:
            img, target = self.data[index], self.labels[index]
            img = np.transpose(img, (1, 2, 0))  # Transpose image channels to convert into CIFAR format
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class DataTransformationMNIST(cts.MNIST[3]):
    def __init__(self, root, indeces, transform=None, target_transform=None, download=True, train=True):
        # Accessing CIFAR10 from torchvision
        super().__init__(root, transform=transform, target_transform=target_transform, download=download, train=train)
        if indeces is not None:
            self.data = self.data[indeces]
            self.targets = np.array(self.targets)[indeces]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array(np.repeat(img[:, :, np.newaxis], 3, axis=2))
        img = Image.fromarray(img)
        img = img.resize((32, 32))  # Resize for 32x32 images
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
            augment(cta, False),
            cutout_strong(level=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weakly_augment = self.weakly(x)
        strongly_augment = self.strongly(x)
        return weakly_augment, strongly_augment


# -----LOADING DATA----- #
def load_dataset(dataset):
    labels = None
    if dataset == 'CIFAR-10':
        raw_data = datasets.CIFAR10(cts.DATASET[5], train=True, download=True)
        labels = np.array(raw_data.targets)
    elif dataset == 'MNIST':
        raw_data = datasets.MNIST(cts.DATASET[5], train=True, download=True)
        labels = np.array(raw_data.targets)
    elif dataset == 'SVHN':
        raw_data = datasets.SVHN(cts.DATASET[5], split='train', download=True)
        labels = np.array(raw_data.labels)
    else:
        print("Wrong dataset name")
        exit(0)

    test_data = None
    if dataset == 'CIFAR-10':
        test_data = datasets.CIFAR10(cts.DATASET[5], train=False,
                                     transform=tensor_normalizer(mean=cts.DATASET[1], std=cts.DATASET[2]),
                                     download=True)
    elif dataset == 'MNIST':
        test_data = datasets.MNIST(cts.DATASET[5], train=False,
                                   transform=tensor_normalizer(mean=cts.DATASET[1], std=cts.DATASET[2]),
                                   download=True)
    elif dataset == 'SVHN':
        test_data = datasets.SVHN(cts.DATASET[5], split='test',
                                  transform=tensor_normalizer(mean=cts.DATASET[1], std=cts.DATASET[2]),
                                  download=True)
    else:
        exit(0)

    return labels, test_data


def dataset_loader(dataset, num_labeled, balanced_split=True, unbalance=0, unbalanced_proportion=1,
                   sample_proportion=1):
    labels, test_data = load_dataset(dataset)

    labeled_indeces, unlabeled_indeces = split_labeled_unlabeled(
        num_labeled,
        n_classes=10,
        labels=labels,
        balanced_split=balanced_split,
        unbalance=unbalance,
        unbalanced_proportion=unbalanced_proportion,
        sample_proportion=sample_proportion
    )

    return labeled_indeces, unlabeled_indeces, test_data

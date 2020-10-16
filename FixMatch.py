import random
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import CTAugment as ctaug
from data.data_loader import *
import torch.optim as optim
from train import *
from exp_moving_avg import EMA
from WideResNet_PyTorch.src import WideResNet as wrn
import torchvision

DATA_ROOT = './data'


# -----SET VARIABLES----- #
def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)


# -----DEFINE FUNCTIONS----- #
def cyclic_learning_rate_with_warmup(warmup_steps, epoch, total_training_epochs):
    # If you don't achieve the number of warmup steps, don't update
    if epoch < warmup_steps:
        new_learning_rate_multiplicative = lambda x: 1
    else:  # Once you surpass the number of warmup steps,
        # you should decay they learning rate close zero in a cosine manner
        new_learning_rate_multiplicative = lambda x: np.cos(7. / 16. * np.pi * (epoch / total_training_epochs))
    # Update learning rate scheduler
    return new_learning_rate_multiplicative


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # Pre-defining mean and std for the datasets to reduce computational time
    CIFAR10_mean = (0.4914, 0.4822, 0.4465)
    CIFAR10_std = (0.2471, 0.2435, 0.2616)
    MNIST_mean = 0.1307
    MNIST_std = 0.3081
    SVHN_mean = (0.4377, 0.4438, 0.4728)
    SVHN_std = (0.1980, 0.2010, 0.1970)

    set_seed(42)
    n_labeled_data = 4000  # We will train with 4000 labeled data to avoid computing many times the CTAugment
    B = 64  # B from the paper, i.e. number of labeled examples per batch.
    mu = 7  # Hyperparam of Fixmatch determining the relative number of unlabeled examples w.r.t. B * mu
    unlabeled_batch_size = B * mu
    warmup_steps = 10  # Define number of warmup steps to avoid premature cyclic learning
    initial_learning_rate = 0.3  # Small learning rate, which with cyclic decay will tend to zero
    momentum = 0.9  # Momentum to access the Stochastic Gradient Descent
    nesterov_factor = False  # They found that the nesterov hyperparm wasn't necessary to achieve errors below 5%
    pseudo_label_threshold = 0.95  # Threshold to guarantee confidence on the model
    total_training_epochs = 2 ** 10  # Number of training epochs, without early stopping (assuming the model
    # expects to see 2^26 images during the whole training)
    initial_training_epoch = 0  # Start the training epoch from zero
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # Create device to perform computations in GPU (if available)

    # -----Define WideResNet Architecture-----#
    wrn_depth = 28
    wrn_width = 2
    n_classes = 10
    strides = [1, 1, 2, 2]
    channels = 3  # Maybe this has to be changed in order to support grayscale

    # -----START MODEL----- #

    # show images of unlabeled data
    '''
    for i, (images, labels) in enumerate(unlabeled_train_data):
        print(i)
        print('Number of labels ', len(labels))
        print('Weakly and Strongly? ', len(images))
        print('Number of image in weakly augmented ', len(images[0]))
        # imshow(torchvision.utils.make_grid(images[0][0]))
        break
    '''
    # Create Wide - ResNet based on the data set characteristics
    model = wrn.WideResNet(d=wrn_depth, k=wrn_width, n_classes=n_classes, input_features=channels,
                           output_features=16, strides=strides)

    # Define Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, nesterov=nesterov_factor)

    # Create scheduler that will take charge of warming up and performing learning rate decay
    # LambdaLR: Sets the learning rate of each parameter group to the initial lr times a given function.
    # (Pytorch documentation)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps,
                                                                                         initial_training_epoch,
                                                                                         total_training_epochs))

    # Define exponential moving avarage of the parameters with 0.999 weight decay
    exp_moving_avg = EMA(model.parameters(), decay=0.999)

    # Analyze the training process
    acc_model = []
    acc_ema = []
    # Query datasets
    labeled_indeces, unlabeled_indeces, test_data = load_cifar10(DATA_ROOT, n_labeled_data)

    # Define CTA augmentation
    cta = ctaug.CTAugment(depth=2, t=0.8, ro=0.99)

    for epoch in range(total_training_epochs):
        # Apply transformations
        labeled_dataset, unlabeled_dataset, train_label_cta = applyTransformations(DATA_ROOT,
                                                                                   labeled_indeces,
                                                                                   unlabeled_indeces,
                                                                                   CIFAR10_mean,
                                                                                   CIFAR10_std,
                                                                                   cta)

        # Load datasets
        labeled_train_data = DataLoader(labeled_dataset, sampler=RandomSampler(labeled_dataset), batch_size=B,
                                        num_workers=0,
                                        drop_last=True)
        unlabeled_train_data = DataLoader(unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
                                          batch_size=unlabeled_batch_size, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=B, num_workers=0)

        labeled_train_cta_data = DataLoader(train_label_cta, sampler=RandomSampler(train_label_cta), batch_size=n_labeled_data,
                                            num_workers=0,
                                            drop_last=True)

        # Update of CTA
        model.to(device)
        cta.update_CTA(model, labeled_train_cta_data, device)



        # Initialize training
        model.zero_grad()
        model.train()

        # Current learning rate to compute the loss combination
        lambda_unsupervised = 1

        # Train model, update weights per epoch based on the combination of labeled and unlabeled losses
        semi_supervised_loss, supervised_loss, unsupervised_loss, supervised_loss_list, unsupervised_loss_list \
            = train_fixmatch(model,
                             device,
                             labeled_train_data,
                             unlabeled_train_data,
                             lambda_unsupervised,
                             B,
                             unlabeled_batch_size,
                             pseudo_label_threshold
                             )
        # Update the weights
        semi_supervised_loss.backward()

        # Update learning rate, optimizer (SGD), and exponential moving average parameters
        optimizer.step()
        scheduler.step()
        exp_moving_avg.update(model.parameters())

        # Test and compute the accuracy for the current model and exponential moving average
        acc_model_tmp, acc_ema_tmp = test_fixmatch(exp_moving_avg, model, test_loader, B, device)

        # Stack learning process
        acc_model.append(acc_model_tmp)
        acc_ema.append(acc_ema_tmp)
        print('Accuracy of the model', acc_model[-1])
        print('Accuracy of ema', acc_ema[-1])


if __name__ == "__main__":
    main()

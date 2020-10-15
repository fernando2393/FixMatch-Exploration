import argparse
import logging
import os
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as torch_functional
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import load_cifar10
import torchvision
from train import *
from exp_moving_avg import EMA
from WideResNet_PyTorch.src import WideResNet


###### SET VARIABLES ######
def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed = 42
set_seed(seed)
DATA_ROOT = './data'
n_labeled_data = 4000 # We will train with 4000 labeled data to avoid computing many times the CTAugment
B = 64 # B from the paper, i.e. number of labeled examples per batch. 
mu = 7 # Hyperparam of Fixmatch determining the relative number of unlabeled examples w.r.t. B * mu
unlabeled_batch_size = B * mu
warmup_steps = 5 # Define number of warmup steps to avoid premature cyclic learning
initial_learning_rate = 0.3 # Small learning rate, which with cyclic decay will tend to zero
momentum = 0.9 # Momentum to access the Stochastic Gradient Descent
nesterov_factor = False # They found that the nesterov hyperparm wasn't necessary to achieve errors below 5% 
pseudo_label_threshold = 0.95 # Threshold to guarantee confidence on the model
total_training_epochs = 5 # Number of training epochs, without early stopping
initial_training_epoch = 0 # Start the training epoch from zero
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Create device to perform computations in GPU (if available)

###### Define WideResNet Architecture
wrn_depth = 28
wrn_width = 2
n_classes = 10
strides = [1, 1, 2, 2]
channels = 3 # Maybe this has to be changed in order to support grayscale 

###### DEFINE FUNCTIONS ######
def cyclic_learning_rate_with_warmup(warmup_steps, epoch, total_training_epochs):
    # If you don't achieve the number of warmup steps, don't update
    if epoch < warmup_steps:
        new_learning_rate_multiplicative = 1
    else: # Once you surpass the number of warmup steps, you should decay they learning rate close zero in a cosine manner
        new_learning_rate_multiplicative = np.cos(7./16. * np.pi * (epoch / total_training_epochs))
    # Update learning rate scheduler
    return new_learning_rate_multiplicative

###### START MODEL ######

# Query datasets
labeled_dataset, unlabeled_dataset, test_dataset = load_cifar10(DATA_ROOT, n_labeled_data)

# Load datasets
labeled_train_data = DataLoader(labeled_dataset, sampler = RandomSampler(labeled_dataset), batch_size= B, num_workers= 0, drop_last=True)
unlabeled_train_data = DataLoader(unlabeled_dataset, sampler = RandomSampler(unlabeled_dataset), batch_size = unlabeled_batch_size, num_workers = 0, drop_last=True)
test_loader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = B, num_workers= 0)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# show images of unlabeled data
for i, (images, labels) in enumerate(unlabeled_train_data):
    print(i)
    print('Number of labels ', len(labels))
    print('Weakly and Strongly? ', len(images))
    print('Number of image in weakly augmented ', len(images[0]))
    imshow(torchvision.utils.make_grid(images[0][0]))
    break
    

# Create Wide - ResNet based on the data set characteristics
model = WideResNet(d=wrn_depth, k=wrn_width, n_classes=n_classes, input_features=channels, output_features=n_classes, strides=strides)

# Define Stochastic Gradient Descent 
optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, nesterov=nesterov_factor)

# Create scheduler that will take charge of warming up and performing learning rate decay
# LambdaLR: Sets the learning rate of each parameter group to the initial lr times a given function. (Pytorch documentation)
scheduler = LambdaLR(optimizer, lr_lambda = cyclic_learning_rate_with_warmup(warmup_steps, initial_training_epoch, total_training_epochs))

# Define exponential moving avarage of the parameters with 0.999 weight decay
exp_moving_avg = EMA(model.parameters(), decay=0.999)

# Analyze the training process
acc_model = []
acc_ema = []

for epoch in range(total_training_epochs):
    # Initialize training
    model.zero_grad().to(device)
    model.train()

    # Current learning rate to compute the loss combination
    lr = cyclic_learning_rate_with_warmup(warmup_steps, epoch, total_training_epochs)

    # Train model, update weights per epoch based on the combination of labeled and unlabeled losses
    semi_supervised_loss, supervised_loss, unsupervised_loss = train_fixmatch(
                                                                            model,
                                                                            device,
                                                                            labeled_train_data,
                                                                            unlabeled_train_data,
                                                                            lr,
                                                                            B,
                                                                            unlabeled_batch_size,
                                                                            pseudo_label_threshold
                                                                            )

    # Update the weights
    semi_supervised_loss.backward()

    # Update learning rate, optimizer (SGD), and exponential moving average parameters
    scheduler.step()
    optimizer.step()
    exp_moving_avg.update(model.parameters())

    # Test and compute the accuracy for the current model and exponetial moving average
    acc_model_tmp, acc_ema_tmp = test_fixmatch(exp_moving_avg, model, test_loader, B)

    # Stack learning process
    acc_model.extend(acc_model_tmp)
    acc_ema.extend(acc_ema_tmp)
    print('Accuracy of the model', acc_model[-1])
    print('Accuracy of ema', acc_ema[-1])



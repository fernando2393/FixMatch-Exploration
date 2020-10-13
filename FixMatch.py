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


###### SET VARIABLES ######
np.random.seed(42) # Define randomness for reproducibility for numpy
torch.manual_seed(42) # Define randomness for reproducibility for pytorch
DATA_ROOT = './data'
n_labeled_data = 4000 # We will train with 4000 labeled data to avoid computing many times the CTAugment
B = 64 # B from the paper, i.e. number of labeled examples per batch. 
mu = 7 # Hyperparam of Fixmatch determining the relative number of unlabeled examples w.r.t. B * mu
warmup_steps = 5 # Define number of warmup steps to avoid premature cyclic learning
initial_learning_rate = 0.3 # Small learning rate, which with cyclic decay will tend to zero
momentum = 0.9 # Momentum to access the Stochastic Gradient Descent
nesterov_factor = False # They found that the nesterov hyperparm wasn't necessary to achieve errors below 5% 
pseudo_label_threshold = 0.95 # Threshold to guarantee confidence on the model
total_training_epochs = 5 # Number of training epochs, without early stopping
initial_training_epoch = 0 # Start the training epoch from zero

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
labeled_trainloader = DataLoader(labeled_dataset, sampler = RandomSampler(labeled_dataset), batch_size= B, num_workers= 0, drop_last=True)
unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler = RandomSampler(unlabeled_dataset), batch_size = B*mu, num_workers = 0, drop_last=True)
test_loader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size= B, num_workers= 0)

'''def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# show images of unlabeled data
for i, (images, labels) in enumerate(unlabeled_trainloader):
    print(i)
    print('Number of labels ', len(labels))
    print('Weakly and Strongly? ', len(images))
    print('Number of image in weakly augmented ', len(images[0]))
    imshow(torchvision.utils.make_grid(images[0][0]))
    break
    '''

# Create Wide - ResNet based on the data set characteristics
model = None

# Define Stochastic Gradient Descent 
optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, nesterov=nesterov_factor)

# Create scheduler that will take charge of warming up and learning rate decay
scheduler = LambdaLR(optimizer, lr_lambda = cyclic_learning_rate_with_warmup(warmup_steps, initial_training_epoch, total_training_epochs))

# Initialize training
model.zero_grad()
for epoch in range(total_training_epochs):
    # Train model, update weights in each batch based on the combination of labeled and unlabeled losses
    model = train_fixmatch()

    # Test
    accuracy = test_fixmatch()




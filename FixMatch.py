import argparse
import logging
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as torch_functional
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from WideResNet import *


if __name__ == "main":
    DATA_ROOT = './data'      
    B = 64 # B from the paper, i.e. number of labeled examples per batch. 
    mu = 7 # Hyperparam of Fixmatch determining the relative number of unlabeled examples w.r.t. B * mu
    n_labeled_data = 4000 # We will train with 4000 labeled data to avoid computing many times the CTAugment


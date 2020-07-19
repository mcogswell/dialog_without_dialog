import os
import gc
import random
import pprint
from six.moves import range
from time import gmtime, strftime
from timeit import default_timer as timer
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import options
from misc.dataloader import VQAPoolDataset
import misc.utilities as utils
from misc.eval_questioner import evalQBot
import torch.nn.functional as F
from torch.distributions.categorical import Categorical 

import numpy as np
import pdb

params = options.readCommandLine()
data_params = options.data_params(params)
data_params['useRedis'] = 1
# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
#if params['useGPU']:
#    torch.cuda.manual_seed_all(params['randomSeed'])

splits = ['train', 'val1', 'val2']

# Setup dataloader
dataset = VQAPoolDataset(data_params, splits)
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=3,
    pin_memory=False)


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

numIterPerEpoch = len(dataloader)
start_t = timer()

for epochId, idx, batch in batch_iter(dataloader):
    iterId = idx + (epochId * numIterPerEpoch)
    epochFrac = iterId / numIterPerEpoch

    end_t = timer()  # Keeping track of iteration(s) time
    timeStamp = strftime('%a %d %b %y %X', gmtime())
    log_line = f'[{timeStamp}][Ep: {epochFrac:.2f}][Iter: {iterId}][Time: {end_t - start_t:5.2f}s]'
    print(log_line)
    start_t = end_t

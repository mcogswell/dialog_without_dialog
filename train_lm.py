import os
import gc
import random
import pprint
import warnings
from six.moves import range
from time import gmtime, strftime
from timeit import default_timer as timer
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

import options
from misc.question_dataloader import VQAQuestionDataset
import misc.utilities as utils
from misc.eval_questioner import evalQBot

import torch.nn.functional as F
from torch.distributions.categorical import Categorical 

import numpy as np
import pdb

params = options.readCommandLine()
data_params = options.data_params(params)
model_params = options.model_params(params)
# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
dataset = VQAQuestionDataset(data_params, ['train', 'val1'])

# Params to transfer from dataset
transfer = ['vocabSize', 'startToken', 'endToken', 'unkToken']
for key in transfer:
    if hasattr(dataset, key):
        model_params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
os.makedirs(params['savePath'], exist_ok=True)

# load model
lm = utils.loadModel(model_params, 'lm', params['useGPU'])

dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=True,
    num_workers=params['numWorkers'],
    drop_last=True,
    pin_memory=True)

viz = utils.TBlogger('logs', params['saveName'])

lRate = params['learningRate']
startIterID = 0
parameters = list(lm.parameters())
optimizer = optim.Adam(parameters, lr=lRate, amsgrad=True)
assert not params['continue']
runningLoss = None

ce_criterion = nn.CrossEntropyLoss()
ce_criterion_nored = nn.CrossEntropyLoss(reduction='none')
# bce_criterion = nn.BCEWithLogitsLoss()

numIterPerEpoch = len(dataloader)
print('\n%d iter per epoch.' % numIterPerEpoch)

#---------------------------------------------------------------------------------------
# Logging, evaluation and checkpoint wrappers
#---------------------------------------------------------------------------------------

def evalLM(params, lm, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=params['batchSize'],
        shuffle=False,
        num_workers=params['numWorkers'],
        pin_memory=True)

    total_loss = 0
    total_nll = 0
    count = 0
    start_t = timer()
    numBatches = len(dataloader)

    for idx, batch in enumerate(dataloader):
        if params['useGPU']:
            batch = {
                key: v.cuda()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        batch_size = batch['ques'].shape[0]
        count += batch_size
        nll = 0

        # forward
        ques_log_probs = lm(None, None, batch['ques'])
        nll += utils.maskedNll(ques_log_probs, batch['ques']) * batch_size
        loss = nll
        total_loss += loss
        total_nll += nll

        if idx % 500 == 0:
            end_t = timer()
            print(f"[Qbot] Evaluating split '{dataset.split}' [{idx+1}/{numBatches}]\t" +
                  f"Time: {end_t - start_t:5.2f}s")
            start_t = end_t

    return {
        'nll': total_nll / count,
        'loss': total_loss / count,
        'count': count,
    }


def evaluate(lm, epochId, params, dataset, **kwargs):
    if idx != 0:
        return

    torch.set_grad_enabled(False)
    lm.eval()

    # Mapping iteration count to epoch count
    viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')
    print('Performing validation...', end='')
    start_t = timer()
    evalResult = evalLM(params, lm, dataset)
    end_t = timer()
    print(f'done in {end_t - start_t:5.2f}s')
    print('Validation:')
    viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')
    val_metrics = ['loss', 'nll']
    for metric in val_metrics:
        value = evalResult[metric]
        try:
            value = float(value)
            viz.linePlot(
                epochId, value, 'val', metric, xlabel='Epochs')
        except:
            pass
    torch.set_grad_enabled(True)


def checkpoint(params, final=False, **kwargs):
    if idx != 0 and not final:
        return

    # Save the model
    params['ckpt_iterid'] = iterId
    params['ckpt_lRate'] = lRate
    saveFile = os.path.join(params['savePath'],
                            'lm_ep_%d.vd' % epochId)
    print('Saving model: ' + saveFile)
    utils.saveModel(lm, optimizer, saveFile, params)


def log_stuff(iterId, epochFrac, timer_info, loss, **kwargs):
    # Print every now and then
    if iterId % 100 == 0 and iterId != 0:
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        log_line = (f'[{timeStamp}][Ep: {epochFrac:.2f}][Iter: {iterId}]' +
            f'[Load Time: {timer_info["load_diff"]/timer_info["log_N"]:5.2f}s]' +
            f'[Compute Time: {timer_info["compute_diff"]/timer_info["log_N"]:5.2f}s]' +
            f'[Loss: {loss:.5g}]')
        print(log_line)

        timer_info['load_diff'] = 0
        timer_info['compute_diff'] = 0
        timer_info['log_N'] = 0

        # Update line plots
        viz.linePlot(iterId, float(loss), 'train', 'total loss')
        viz.linePlot(iterId, float(lRate), 'train', 'learning rate')


#---------------------------------------------------------------------------------------
# Training
#---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

# track load time
timer_info = {
    'load_diff': 0,
    'compute_diff': 0,
    'log_N': 0,
}
start_t = timer()

for epochId, idx, batch in batch_iter(dataloader):
    # track load time
    end_t = timer()
    timer_info['load_diff'] += end_t - start_t
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epochFrac = iterId / numIterPerEpoch

    # Evaluate and checkpoint
    checkpoint(**locals())
    evaluate(**locals())

    # Train
    start_t = timer()
    if params['useGPU']:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
                                    else v for key, v in batch.items()}
    lm.train()
    timer_info['log_N'] += 1
    nll = 0
    batch_size = batch['ques'].shape[0]

    # forward
    ques_log_probs = lm(None, None, batch['ques'])
    nll += utils.maskedNll(ques_log_probs, batch['ques'])
    loss = nll

    # step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    end_t = timer()
    timer_info['compute_diff'] += end_t - start_t
    log_stuff(**locals())
    start_t = timer()


checkpoint(final=True, **locals())


print('Finished')

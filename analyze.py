import os.path as pth
from itertools import islice

import torch
from torch.utils.data import DataLoader

import pandas as pd
import joblib

import options
import misc.utilities as utils
from misc.dataloader import VQAPoolDataset
from misc.cub_dataloader import CUBPoolDataset
from misc.awa_dataloader import AWAPoolDataset
from misc.eval_questioner import evalQBot

def sample_batch(batch, n_samples=10):
    sbatch = {}
    for k in batch:
        shape = batch[k].shape
        sbatch[k] = batch[k].unsqueeze(0) \
                            .expand(n_samples, *([-1] * len(shape))) \
                            .reshape((n_samples * shape[0],) + shape[1:])
    return sbatch

def unflat_sample_batch(sbatch, n_samples=10):
    batch = {}
    for k in sbatch:
        shape = sbatch[k].shape
        assert shape[0] % n_samples == 0
        batch[k] = sbatch[k].reshape((n_samples, shape[0] // n_samples) + shape[1:])                            
    return batch

def guess_epochs(params, agent='qbot'):
    qbot_file = pth.join(params['savePath'], '{}_ep_{}.vd')
    for ep in reversed(range(params['numEpochs'] + 1)):
        curr = qbot_file.format(agent, ep)
        if pth.exists(curr):
            return [ep]
    if agent == 'qbot':
        raise Exception('No viable checkpoints found for {}'.format(params['savePath']))
    elif agent == 'abot':
        # use astartFrom
        return [-1]

def load_models(params):
    # iteratively load models that need to be evaluated
    qepochs = guess_epochs(params, 'qbot')
    aepochs = guess_epochs(params, 'abot')
    gpu = params['useGPU']
    for qep, aep in zip(qepochs, aepochs):
        exp = params['saveName']
        qbot_file = pth.join(params['savePath'], 'qbot_ep_{}.vd'.format(qep))
        qbot = utils.loadModelFromFile(qbot_file, agent='qbot', gpu=gpu)
        qbot.eval()
        if aep == -1:
            abot_file = params['astartFrom']
        else:
            abot_file = pth.join(params['savePath'], 'abot_ep_{}.vd'.format(aep))
        abot = utils.loadModelFromFile(abot_file, agent='abot', gpu=gpu)
        abot.eval()
        yield qbot, abot, exp, qbot_file, abot_file, qep, aep

def main():
    params = options.readCommandLine()
    # memory for unique samples is a concern; use smaller batch sizes
    n_samples = min(50, params['batchSize'])
    unique_batch_size = params['batchSize'] // n_samples
    unique_examples = 300
    unique_num_batches = (unique_examples - 1) // unique_batch_size + 1
    z_sources = ['policy', 'encoder', 'prior']
    # z, dec
    inferences = [('greedy', 'greedy'), ('sample', 'greedy'), ('greedy', 'sample'), ('sample', 'sample')]

    data_params = options.data_params(params)
    if params['dataset'] == 'VQA':
        dataset = VQAPoolDataset(data_params, ['train', 'val1', 'val2'])
        val_split = {
            'val1': 'val1',
            'val2': 'val2',
            'val': 'val1',
            'test': 'val2',
        }[params['evalSplit']]
    elif params['dataset'] == 'CUB':
        dataset = CUBPoolDataset(data_params, ['train', 'val', 'test'], load_vis_info=True)
        val_split = params['evalSplit']
    elif params['dataset'] == 'AWA':
        dataset = AWAPoolDataset(data_params, ['train', 'val', 'test'], load_vis_info=True)
        val_split = params['evalSplit']
    dataset.split = val_split

    #unique_records = []
    eval_records = []
    for qbot, abot, q_exp, qbf, abf, qep, aep in load_models(params):

        # evaluate task performance
        params_ = params.copy()
        params_['trainMode'] = 'fine-qbot'
        print('evaluating {}'.format(str(dataset.params)))
        result = evalQBot(params_, qbot, dataset, val_split, aBot=abot,
                                exampleLimit=params['evalLimit'],
                                do_perplexity=True,
                                do_relevance=True,
                                do_diversity=True)
        result['abot_file'] = abf
        result['qbot_file'] = qbf
        result['qbot'] = q_exp
        result['qepoch'] = qep
        result['aepoch'] = aep
        result['poolType'] = dataset.poolType if dataset.name == 'VQA' else dataset.pool_type
        result['poolSize'] = dataset.poolSize if dataset.name == 'VQA' else dataset.pool_size
        result['randQues'] = dataset.randQues if dataset.name == 'VQA' else False
        if 'lang_result' in result and 'byRound' in result['lang_result']:
            for d in result['lang_result']['byRound']:
                del d['imgToEval']
        else:
            result['lang_result'] = {}
        eval_records.append(result)

    out_prefix = 'pt-{}_ps-{}'.format(params['poolType'], params['poolSize'])

    #unique_df = pd.DataFrame(unique_records)
    eval_df = pd.DataFrame(eval_records)

    out_file = pth.join(params['savePath'], '{}.joblib'.format(params['evalSaveName']))
    joblib.dump({
        #'unique_df': unique_df,
        'eval_df': eval_df,
    }, out_file, compress=True)


if __name__ == '__main__':
    main()

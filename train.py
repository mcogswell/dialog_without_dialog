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
from misc.dataloader import VQAPoolDataset
from misc.cub_dataloader import CUBPoolDataset
from misc.awa_dataloader import AWAPoolDataset
import misc.utilities as utils
from misc.eval_questioner import evalQBot
from models.vqg import VQGModel

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
if params.get('dataset', 'VQA') == 'VQA':
    splits = ['train', 'val1', 'val2']
    val_split = 'val1'
    dataset = VQAPoolDataset(data_params, splits)
elif params['dataset'] == 'CUB':
    assert params['poolType'] == 'random'
    splits = ['train', 'val', 'test']
    val_split = 'val'
    dataset = CUBPoolDataset(data_params, splits)
elif params['dataset'] == 'AWA':
    assert params['poolType'] == 'random'
    splits = ['train', 'val', 'test']
    val_split = 'val'
    dataset = AWAPoolDataset(data_params, splits)
# used for evaluation even when not mixing
if params['trainMode'] == 'fine-qbot' and params.get('mixing', 0) != 0:
    mixing_data_params = data_params.copy()
    mixing_data_params['poolType'] = 'contrast'
    mixing_data_params['poolSize'] = 2
    mixing_dataset = VQAPoolDataset(data_params, splits)
    pre_eval_dataset = mixing_dataset
else:
    mixing_dataset = None
    pre_eval_dataset = dataset

# Params to transfer from dataset
transfer = ['vocabSize', 'startToken', 'endToken', 'unkToken', 'num_ans_candidates']
for key in transfer:
    if hasattr(dataset, key):
        model_params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
os.makedirs(params['savePath'], exist_ok=True)

# Create/Load Modules
parameters = []
aBot = None
qBot = None

# load qbot
if params['qstartFrom']:
    qBot = utils.loadModelFromFile(params['qstartFrom'], 'qbot',
                                   params['useGPU'], params['speakerStartFrom'])
    qBot.train()
elif 'abot' not in params['trainMode']:
    qBot = utils.loadModel(model_params, 'qbot',
                           params['useGPU'], params['speakerStartFrom'])

if params['ewcLossCoeff'] > 0:
    prior_qBot_parameters = {n: p.clone().detach() for n, p in qBot.named_parameters()}
def compute_ewc_loss(scores, qBot, prior_qBot_parameters):
    N = scores.shape[0]
    ewc = 0
    # NOTE: This single forward pass with many backwards deals with modules
    # that use batch statistics like BatchNorm. The loss value should probably
    # depend on the entire batch. That doesn't happen if examples are forwarded
    # individually, as in some other implementations.
    for i in range(N):
        print(i)
        import pdb; pdb.set_trace()
        optimizer.zero_grad()
        scores[i].backward(retain_graph=True)
        for n, p in qBot.named_parameters():
            # the 2nd can happen for unused encoder parameters
            if not p.requires_grad or p.grad is None:
                continue
            Fi = p.grad.pow(2)
            pp = prior_qBot_parameters[n]
            ewc += (Fi * (p - pp)**2).sum()
    return ewc / 2

# load abot
if params['astartFrom']:
    aBot = utils.loadModelFromFile(params['astartFrom'], 'abot', params['useGPU'])
    aBot.train()
elif 'abot' in params['trainMode'] or 'fine' in params['trainMode']:
    aBot = utils.loadModel(model_params, 'abot', params['useGPU'])

vqg = None
# load vqg model
if (params.get('vqgstartFrom') is not None or
    params.get('qcycleLossCoeff', 0) != 0):
    # TODO: proper checkpointing and loading
    # TODO: proper decoder sharing.... what's the right way to do that?
    vqg = VQGModel(model_params, qBot.decoder)
    vqg.cuda()
    aBot.tracking = True

def fix_params(freeze_mode):
    # define some parameter groups
    decoder_params = ['z2hdec', 'gumbel_codebook', 'decoder']
    predict_params = ['predict_']
    dialog_params = ['dialogRNN']
    policy_pt1_params = ['ansEmebed', 'quesStartEmbed', 'quesEncoder',
                         'ctx_coder', 'query_embed']
    policy_pt2_params = ['hc2z', 'henc2z', 'encoder']
    policy_params = policy_pt1_params + policy_pt2_params
    # decide which params
    fix_params_names = ['speaker'] # speaker is always fixed
    if freeze_mode == 'decoder':
        fix_params_names += decoder_params
    elif freeze_mode == 'predict':
        fix_params_names += decoder_params + predict_params
    elif freeze_mode == 'all_but_policy':
        fix_params_names += decoder_params + predict_params + dialog_params
    elif freeze_mode == 'all_but_ctx':
        fix_params_names += decoder_params + predict_params + dialog_params + policy_pt2_params
    elif freeze_mode == 'policy':
        fix_params_names += decoder_params + policy_params
    # actually fix them
    fix_params_names
    exclude_params = []
    include_params = []
    inc_tensors = []
    if qBot is not None:
        for key, value in dict(qBot.named_parameters()).items():
            if np.sum([name in key for name in fix_params_names]):
                value.requires_grad = False
                exclude_params.append(key)
            else:
                value.requires_grad = True
                include_params.append(key)
                inc_tensors.append(value)
    return exclude_params, include_params, inc_tensors

# decide which parameters to optimize
if 'fine-qbot' in params['trainMode']:
    exclude_params, include_params, inc_tensors = fix_params(params['freezeMode'])
    print(f'Params exclude from Training {exclude_params}')
    print(f'Params included in Training {include_params}')
    parameters = inc_tensors
    if params.get('freezeMode2') is not None:
        assert 'freezeMode2Freq' in params
        exclude_params2, include_params2, inc_tensors2 = fix_params(params['freezeMode2'])
        print(f'Freeze Mode 2 Params exclude from Training {exclude_params2}')
        print(f'Freeze Mode 2 Params included in Training {include_params2}')
        parameters = list(set(inc_tensors + inc_tensors2))
elif 'qbot' in params['trainMode']:
    if params['freezeMode'] != 'none':
        warnings.warn('Attempting to freeze some parameters during initial '
                      'pre-training. Check to be sure this is what you want.')
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))
if 'abot' in params['trainMode']:
    parameters.extend(filter(lambda p: p.requires_grad, aBot.parameters()))
if vqg is not None:
    assert params.get('freezeMode2') is None
    if params['freezeMode'] == 'all_but_ctx':
        parameters = []
    parameters.extend(filter(lambda p: p.requires_grad, vqg.parameters()))


dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=True,
    num_workers=params['numWorkers'],
    drop_last=True,
    pin_memory=True)
if params.get('mixing', 0) != 0:
    mixing_dataset.split = 'train'
    mixing_dataloader = DataLoader(
        mixing_dataset,
        batch_size=params['batchSize'],
        shuffle=True,
        num_workers=params['numWorkers'],
        drop_last=True,
        pin_memory=True)
else:
    mixing_dataloader = None

viz = utils.TBlogger('logs', params['saveName'])

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate, amsgrad=True)
if params['scheduler'] == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                patience=params['lrPatience'],
                                cooldown=params['lrCooldown'],
                                factor=params['lrDecayRate'],
                                min_lr=params['minLRate'])
else:
    scheduler = None
assert not params['continue']
runningLoss = None

ce_criterion = nn.CrossEntropyLoss()
ce_criterion_nored = nn.CrossEntropyLoss(reduction='none')
# bce_criterion = nn.BCEWithLogitsLoss()

numIterPerEpoch = len(dataloader)
print('\n%d iter per epoch.' % numIterPerEpoch)

#---------------------------------------------------------------------------------------
# Training
#---------------------------------------------------------------------------------------

def batch_iter(dataloader, mixing_dataloader=None):
    mode = params['trainMode']
    if mixing_dataloader is not None:
        mixing_iter = iter(mixing_dataloader)
    freeze_mode1 = params['freezeMode']
    freeze_mode2 = params.get('freezeMode2')
    freeze2_freq = params.get('freezeMode2Freq')
    freeze2_freq = str(0 if freeze2_freq is None else freeze2_freq)
    freeze2_epoch = freeze2_freq.startswith('ep')
    freeze2_freq = int(freeze2_freq[2:] if freeze2_epoch else freeze2_freq)
    for epochId in range(params['numEpochs']):
        if freeze2_epoch:
            freeze_mode = freeze_mode1
            if freeze2_freq and (epochId+1) % freeze2_freq == 0:
                freeze_mode = freeze_mode2
            print(f'epoch {epochId}, mode {freeze_mode}')
        for idx, batch in enumerate(dataloader):
            if not freeze2_epoch:
                freeze_mode = freeze_mode1
                if freeze2_freq and (idx+1) % freeze2_freq == 0:
                    freeze_mode = freeze_mode2
            yield mode, freeze_mode, epochId, idx, batch
            if params.get('mixing', 0) != 0 and idx % params.get('mixing', 0) == 0:
                try:
                    batch = next(mixing_iter)
                except StopIteration:
                    mixing_iter = iter(mixing_dataloader)
                    batch = next(mixing_iter)
                yield 'pre-qbot', freeze_mode, epochId, idx, batch


start_t = timer()
kl_weight_encoder = params['klWeight']
kl_weight_cc = params['klWeightCC']

load_diff = 0
compute_diff = 0
log_N = 0

for trainMode, freeze_mode, epochId, idx, batch in batch_iter(dataloader, mixing_dataloader):
    # track load time
    end_t = timer()
    load_diff += end_t - start_t

    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epochFrac = iterId / numIterPerEpoch
    fix_params(freeze_mode)

    #############################################
    # Evaluate and checkpoint (every epoch)
    # ignore the pre-train iteration of mixed fine-tuning
    if idx == 0 and params['trainMode'] == trainMode: # and iterId != 0:
        # Set eval mode
        if qBot: qBot.eval()
        if aBot: aBot.eval()

        # Mapping iteration count to epoch count
        viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')
        print('Performing validation...')
        print('Validation:')
        evalResult = evalQBot(params, qBot, dataset, val_split, aBot=aBot,
                              exampleLimit=params['evalLimit'], vqg=vqg)
        viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')
        if qBot:
            val_metrics = ['encoder_nll', 'cc_nll', 'predict_loss', 'accuracy',
                           'question_matches', 'question_match_ratio', 'qcycle_loss']
        elif aBot:
            val_metrics = ['abot_accu', 'abot_rel_accu', 'abot_loss']
        for metric in val_metrics:
            value = evalResult[metric]
            try:
                value = float(value)
                viz.linePlot(
                    epochId, value, 'val', metric, xlabel='Epochs')
            except:
                pass
        for i in range(params['maxRounds']):
            if 'R' + str(i) in evalResult:
                viz.linePlot(iterId, float(evalResult['R' + str(i)]), 'val', 'round accuracy ' + str(i))
        if qBot and 'langResult' in evalResult and 'metrics' in evalResult['langResult']:
            for metric in evalResult['langResult']['metrics']:
                for r, result in enumerate(evalResult['langResult']['byRound']):
                    value = result[metric]
                    viz.linePlot(
                        epochId, value, 'val - qBot - lang', 'R' + str(r) + '-' + metric, xlabel='Epochs')
        if qBot and scheduler is not None:
            scheduler.step(evalResult['predict_loss'])
            lRate = min([pg['lr'] for pg in optimizer.param_groups])

        # Also evaluate pre-training metrics
        if params['trainMode'] == 'fine-qbot' and params.get('dataset', 'VQA') == 'VQA':
            print('Evaluating pre-train metrics...')
            preEvalResult = evalQBot(params, qBot, pre_eval_dataset, val_split,
                              aBot=aBot, exampleLimit=params['evalLimit'],
                              trainMode='pre-qbot')
            val_metrics = ['encoder_nll', 'cc_nll', 'predict_loss', 'accuracy']
            for metric in val_metrics:
                value = preEvalResult[metric]
                try:
                    value = float(value)
                    viz.linePlot(
                        epochId, value, 'pre-train val', metric, xlabel='Epochs')
                except:
                    pass
        start_t = timer()

        # Save the model
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate
        if aBot:
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % epochId)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % epochId)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)

    #############################################
    # Train
    # track compute time
    start_t = timer()
    if params['useGPU']:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
                                    else v for key, v in batch.items()}
    if qBot: qBot.train()
    if aBot: aBot.train()
    log_N += 1

    if 'pre' in trainMode:
        numRounds = 1
    else:
        numRounds = params['maxRounds']
    encoder_nll = 0
    encoder_kl = 0
    cc_nll = 0
    cc_kl = 0
    predict_loss = 0
    accuracy = 0
    abot_loss = 0
    abot_rel_loss = 0
    qcycle_loss = 0
    ewc_loss = 0
    round_accuracy = np.zeros(numRounds, dtype=np.float)

    # anneal kl weight
    if params['varType'] == 'cont':
        kl_weight_anneal = utils.kl_anneal_function(params['annealFunction'], iterId, params['k'], params['x0']) 
    else:
        kl_weight_anneal = 1
    # anneal gumbel temperature
    # NOTE: do NOT use trainMode so there's no annealing during fine-tuning
    if params['trainMode'] == 'pre-qbot' and qBot.varType == 'gumbel':
        temp, hard = utils.gumbel_temp_anneal_function(params, iterId, True)
        qBot.temp, qBot.hard = temp, hard

    batch_size = batch['img_pool'].shape[0]
    target_image = batch['target_image']

    if qBot:
        qBot.reset()
        qBot.observe(images=batch['img_pool'])

    # Iterating over the dialog round. 
    # for the pretraining, we only have first round of the dialog.
    for Round in range(numRounds):
        if Round == 0 and 'qbot' in trainMode:
            # observe initial question
            qBot.observe(start_question=True)
            # since we only has 1 round. 
            qBot.observe(start_answer=True)

        if trainMode == 'pre-qbot':
            # observe GT for training.
            qBot.observe(ques=batch['ques'], ques_len=batch['ques_len'], gt_ques=True)

            # Cross Entropy (CE) Loss for Ground Truth Questions   
            encoder_log_probs, encoder_kl, cc_log_probs, cc_kl = qBot()
            encoder_nll += utils.maskedNll(encoder_log_probs, batch['ques'])
            cc_nll += utils.maskedNll(cc_log_probs, batch['ques'])

            # manually roll out part of the 2nd round for image prediction loss
            qBot.observe(ques=batch['ques'], ques_len=batch['ques_len'], gt_ques=False)
            ansRelIdx = torch.ones(batch_size, dtype=torch.uint8)
            qBot.observe(ans=batch['ansIdx'].view(-1), ans_rel=ansRelIdx)

            qBot()
            logit = qBot.predictImage()
            if logit is not None:
                predict_loss += ce_criterion(logit, batch['target_pool'].view(-1))
                _, predict = torch.max(logit, dim=1)
                accuracy += (predict == batch['target_pool'].view(-1)).float().mean()

        if trainMode == 'pre-abot':
            pred, relavent_pred1, relavent_pred2 = aBot(target_image,
                                                        batch['ques'],
                                                        batch['ques_len'],
                                                        batch['rand_ques'],
                                                        batch['rand_ques_len'],
                                                        inference_mode=True)            

            abot_rel_loss += ce_criterion(relavent_pred1, batch['do_stop_target'].view(-1))
            abot_rel_loss += ce_criterion(relavent_pred2, batch['no_stop_target'].view(-1))
            abot_loss += utils.instance_bce_with_logits(pred, batch['ans'])

        if trainMode == 'fine-qbot':
            # decode the question
            ques, ques_len, ques_one_hot = qBot.forwardDecode(
                                            dec_inference='greedy',
                                            z_inference='sample',
                                            z_source=params['zSourceFine'])

            logit = qBot.predictImage()
            predict_loss += ce_criterion_nored(logit, batch['target_pool'].view(-1))

            _, predict = torch.max(logit, dim=1)
            predCorrect = (predict == batch['target_pool'].view(-1)).float()

            # observe the question here.
            qBot.observe(ques=ques, ques_len=ques_len, ques_one_hot=ques_one_hot,
                         gt_ques=False)
            # answer the question here.
            ans, ansRel = aBot(target_image, ques, ques_len, inference_mode=False)
            _, ansIdx = torch.max(ans, dim=1)
            _, ansRelIdx = torch.max(ansRel, dim=1)

            if params.get('qcycleLossCoeff'):
                cycle_log_probs = vqg(ans, ansRel, aBot.v_emb, ques)
                qcycle_loss += utils.maskedNll(cycle_log_probs, ques)

            # to predict the target image, use the latest latent state and the predict answer to select the target images.
            qBot.observe(ans=ansIdx, ans_rel=ansRelIdx)

            round_accuracy[Round] = float(predCorrect.mean())
            accuracy += float((predCorrect).mean())

    if trainMode == 'pre-qbot':
        loss = (predict_loss * params['predictLossCoeff'] +
                encoder_nll +
                encoder_kl * kl_weight_anneal * kl_weight_encoder +
                cc_nll +
                cc_kl * kl_weight_anneal * kl_weight_cc)

    elif trainMode == 'pre-abot':
        loss = abot_loss + abot_rel_loss

    elif trainMode == 'fine-qbot':
        predict_loss /= numRounds
        if params['ewcLossCoeff'] > 0:
            ewc_loss += compute_ewc_loss(-predict_loss, qBot, prior_qBot_parameters)
        predict_loss = predict_loss.mean(dim=0)
        qcycle_loss /= numRounds
        accuracy /= numRounds
        # TODO: ... do we need to add kl terms at fine-tune time??
        loss = (predict_loss * params['predictLossCoeff'] +
                qcycle_loss * params['qcycleLossCoeff'] +
                ewc_loss * params['ewcLossCoeff'])


    optimizer.zero_grad()
    loss.backward()
    if qBot: nn.utils.clip_grad_norm_(qBot.parameters(), 0.5)
    if aBot: nn.utils.clip_grad_norm_(aBot.parameters(), 0.5)
    optimizer.step()

    # Decay learning rate
    if scheduler is None:
        if lRate > params['minLRate']:
            for gId, group in enumerate(optimizer.param_groups):
                optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
            lRate *= params['lrDecayRate']

    #############################################
    # Logging
    # track compute time
    end_t = timer()
    compute_diff += end_t - start_t
    log_for_mixing = (params.get('mixing', 0) != 0 and trainMode == 'pre-qbot')
    # Print every now and then
    if iterId % 10 == 0 and iterId != 0 and not log_for_mixing:
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        log_line = (f'[{timeStamp}][Ep: {epochFrac:.2f}][Iter: {iterId}]' +
            f'[Load Time: {load_diff/log_N:5.2f}s]' +
            f'[Compute Time: {compute_diff/log_N:5.2f}s]' +
            f'[Loss: {loss:.5g}]' +
            f'[PreictLoss: {predict_loss:.5g}]' +
            f'[encoder_NLL: {encoder_nll:.5g}]' +
            f'[cc_NLL: {cc_nll:.5g}]' +
            f'[encoder_kl: {encoder_kl:.5g}]' +
            f'[cc_kl: {cc_kl:.5g}]' +
            f'[kl_weight_encoder: {kl_weight_anneal * kl_weight_encoder:.5g}]' +
            f'[kl_weight_cc: {kl_weight_anneal * kl_weight_cc:.5g}]' +
            f'[accuracy: {accuracy:.2g}]' +
            f'[qcycle_loss: {qcycle_loss:.5g}]' +
            f'[lr: {lRate:.3g}]' +
            f'[tm: {trainMode}]')
        start_t = end_t
        print(log_line)
        print('average_round_accuracy \t',end='')
        for i, racc in enumerate(list(round_accuracy)):
            print(f'R{i} {racc:.2g} ', end='')
        print('')    

        load_diff = 0
        compute_diff = 0
        log_N = 0

        # Update line plots
        viz.linePlot(iterId, float(encoder_nll), 'pre-train', 'encoder_nll')
        viz.linePlot(iterId, float(encoder_kl), 'pre-train', 'encoder_kl')
        viz.linePlot(iterId, float(cc_nll), 'pre-train', 'cc_nll')
        viz.linePlot(iterId, float(cc_kl), 'pre-train', 'cc_kl')
        viz.linePlot(iterId, float(loss), 'train', 'total loss')
        viz.linePlot(iterId, float(qcycle_loss), 'train', 'qcycle_loss')
        viz.linePlot(iterId, float(predict_loss), 'train', 'predict_loss')
        viz.linePlot(iterId, float(accuracy), 'train', 'accuracy')
        viz.linePlot(iterId, float(lRate), 'train', 'learning rate')
        for i in range(len(round_accuracy)):
            viz.linePlot(iterId, float(round_accuracy[i]), 'fine-train', 'round accuracy ' + str(i))
        if hasattr(qBot, 'temp'):
            viz.linePlot(iterId, qBot.temp, 'pre-train', 'gumbel temp')

    if log_for_mixing:
        viz.linePlot(iterId, float(encoder_nll), 'mixed pre-train', 'encoder_nll')
        viz.linePlot(iterId, float(encoder_kl), 'mixed pre-train', 'encoder_kl')
        viz.linePlot(iterId, float(cc_nll), 'mixed pre-train', 'cc_nll')
        viz.linePlot(iterId, float(cc_kl), 'mixed pre-train', 'cc_kl')
        viz.linePlot(iterId, float(loss), 'mixed pre-train', 'total loss')
        viz.linePlot(iterId, float(predict_loss), 'mixed pre-train', 'predict_loss')
        viz.linePlot(iterId, float(accuracy), 'mixed pre-train', 'accuracy')

    # track load time
    start_t = timer()


print('Finished')

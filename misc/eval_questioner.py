import sys
import json
import h5py
import numpy as np
import warnings
from timeit import default_timer as timer

import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F

import options
import misc.utilities as utils
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances
from torch.distributions.categorical import Categorical 

from six.moves import range
import pdb


def count_question_matches(ques1, ques_len1, ques2, ques_len2):
    '''Return the number of questions which match exactly.'''
    assert ques1.shape[0] == ques2.shape[0]
    # [batch_size]
    len_match_mask = ques_len1 == ques_len2
    # [batch_size x max_len]
    word_match_mask = (ques1 == ques2)

    # get lengths
    lens = ques_len1.cpu()
    max_len = torch.tensor(ques1.shape[1] -1)
    lens = torch.min(lens, max_len)
    cumsum = word_match_mask.cumsum(dim=1).cpu()
    # [num_len_match_examples]
    num_word_matches = cumsum.gather(dim=1, index=lens[:, None]-1)[:, 0]
    assert (num_word_matches <= lens).all()
    num_question_matches = (num_word_matches == lens).sum()
    soft_question_matches = (num_word_matches.float() / lens.float()).sum()
    return num_question_matches, soft_question_matches


def evalQBot(params, qBot, dataset, split, aBot=None, exampleLimit=None,
             verbose=0, trainMode=None, vqg=None,
             do_perplexity=False, do_relevance=False, do_diversity=False):
    # basic initialization
    torch.set_grad_enabled(False)    
    ce_criterion = nn.CrossEntropyLoss(reduction='sum')
    bce_criterion = nn.BCEWithLogitsLoss()
    if trainMode is None:
        trainMode = params['trainMode']
    if 'pre' in trainMode:
        numRounds = 1
    else:
        numRounds = params['maxRounds']
    original_split = dataset.split
    dataset.split = split
    if exampleLimit is None:
        numExamples = len(dataset)
    else:
        numExamples = exampleLimit
    if qBot:
        initial_tracking = qBot.tracking

    # dataloader
    numBatches = (numExamples - 1) // params['batchSize'] + 1
    dataloader = DataLoader(
        dataset,
        batch_size=params['batchSize'],
        shuffle=False,
        num_workers=params['numWorkers'],
        pin_memory=True)

    # alternative evaluation metric initialization
    if do_perplexity:
        lm = utils.loadModelFromFile(params['languageModel'], agent='lm', gpu=params['useGPU'])
        ques_pplx = 0
    if do_relevance:
        assert aBot is not None
        max_rel_prob = 0
    if do_diversity:
        all_ques = []

    # tracked variables
    abot_loss = 0
    abot_rel_accu = 0
    abot_accu = 0
    encoder_nll = 0
    cc_nll = 0
    qcycle_loss = 0
    predict_loss = 0
    count = 0
    question_matches = 0
    question_match_ratio = 0
    round_accuracy = np.zeros(numRounds, dtype=np.float)
    accuracy = 0

    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if params['useGPU']:
            batch = {
                key: v.cuda()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }

        batch_size = batch['img_pool'].shape[0]
        count += batch_size
        ansRelIdx = torch.ones(batch_size, dtype=torch.uint8)
        # observe the image
        img_pool = batch['img_pool']
        target_image = batch['target_image']
        if qBot:
            qBot.reset()
            qBot.observe(images=img_pool)

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
                encoder_log_probs, _, cc_log_probs, _ = qBot()
                encoder_nll += utils.maskedNll(encoder_log_probs, batch['ques'])
                cc_nll += utils.maskedNll(cc_log_probs, batch['ques'])

                # manually roll out 2nd round for image prediction
                qBot.observe(ques=batch['ques'], ques_len=batch['ques_len'], gt_ques=False)
                qBot.observe(ans=batch['ansIdx'].view(-1), ans_rel=ansRelIdx)

                qBot()
                logit = qBot.predictImage()
                
                if logit is not None:
                    predict_loss += ce_criterion(logit, batch['target_pool'].view(-1))

                    _, predict = torch.max(logit, dim=1)
                    accuracy += (predict == batch['target_pool'].view(-1)).float().sum()

            if trainMode == 'pre-abot':
                pred, relavent_pred1, relavent_pred2 = aBot(target_image,
                                                        batch['ques'],
                                                        batch['ques_len'],
                                                        batch['rand_ques'],
                                                        batch['rand_ques_len'],
                                                        inference_mode=True)
            
                abot_loss += float(utils.instance_bce_with_logits(pred, batch['ans']))           
                abot_accu += float(utils.compute_score_with_logits(pred, batch['ans']).sum())
                abot_rel_accu += float(torch.sum(torch.max(relavent_pred1, 1)[1] == 1))
                abot_rel_accu += float(torch.sum(torch.max(relavent_pred2, 1)[1] == 0))

            if trainMode == 'fine-qbot':
                # observe GT quesion for z_source == 'encoder'; unused otherwise
                if params['zSourceFine'] == 'encoder':
                    qBot.observe(ques=batch['ques'], ques_len=batch['ques_len'], gt_ques=True)
                # decode the question
                # (ignore one hot questions; don't need gradients)
                ques, ques_len, _ = qBot.forwardDecode(
                                                dec_inference='greedy', 
                                                z_inference='greedy',
                                                z_source=params['zSourceFine'])

                if params['zSourceFine'] == 'encoder':
                    qm = count_question_matches(ques[:, :-1], ques_len,
                                   # batch includes the end token in len
                                   batch['ques'], batch['ques_len'][:, 0] - 1)
                    question_matches += qm[0]
                    question_match_ratio += qm[1]
                    qBot.forwardDecode(dec_inference='greedy', 
                                       z_inference='greedy',
                                       z_source='policy')

                logit = qBot.predictImage()
                predict_loss += ce_criterion(logit, batch['target_pool'].view(-1))

                _, predict = torch.max(logit, dim=1)
                predCorrect = (predict == batch['target_pool'].view(-1)).float()
                # observe the question here.
                qBot.observe(ques=ques, ques_len=ques_len, gt_ques=False)
                # answer the question here.

                ans, ansRel = aBot(target_image, ques, ques_len, inference_mode=False)            
                _, ansIdx = torch.max(ans, dim=1)
                _, ansRelIdx = torch.max(ansRel, dim=1)

                if vqg is not None:
                    cycle_log_probs = vqg(ans, ansRel, aBot.v_emb, ques)
                    qcycle_loss += utils.maskedNll(cycle_log_probs, ques)

                # to predict the target image, use the latest latent state and the predict answer to select the target images.
                qBot.observe(ans=ansIdx, ans_rel=ansRelIdx)
                
                round_accuracy[Round] += float(predCorrect.sum())
                accuracy += float((predCorrect).sum())

                if do_perplexity:
                    log_probs = lm(None, None, ques)
                    log_probs = utils.maskedNll(log_probs, ques,
                                                returnScores=True)
                    entropy = -log_probs / ques_len.to(log_probs)
                    ques_pplx += torch.pow(2, entropy).sum(dim=0)

                if do_relevance:
                    img_pool = batch['img_pool']
                    n_pool = img_pool.shape[1]
                    rel_logit_pool = []
                    for i in range(n_pool):
                        _, rel_logit = aBot(img_pool[:, i], ques, ques_len,
                                            inference_mode=False)
                        rel_logit_pool.append(rel_logit.unsqueeze(1))
                    rel_logit_pool = torch.cat(rel_logit_pool, dim=1)
                    rel_dist_pool = F.softmax(rel_logit_pool, dim=2)
                    rel_prob_pool = rel_dist_pool[:, :, 1]
                    max_rel_prob += rel_prob_pool.max(dim=1)[0].sum(dim=0)

                if do_diversity:
                    gen_ques_str = utils.old_idx_to_str(dataset.ind2word, ques, ques_len, batch['img_id_pool'], 0, [])
                    gen_ques_str = [q[0].strip('<START> ').strip(' <END>') for q in gen_ques_str]
                    all_ques.extend(gen_ques_str)

        if (idx + 1 == numBatches) or (idx % 20 == 0):
            end_t = timer()
            print(f"[Qbot] Evaluating split '{split}' [{idx+1}/{numBatches}]\t" +
                  f"Time: {end_t - start_t:5.2f}s")
            start_t = end_t
        if idx + 1 == numBatches:
            break

    encoder_nll = float(encoder_nll) / count
    cc_nll = float(cc_nll) / count
    qcycle_loss = float(qcycle_loss) / count
    predict_loss = float(predict_loss) / count
    accuracy = float(accuracy) / count
    question_matches = float(question_matches) / count
    question_match_ratio = float(question_match_ratio) / count
    round_accuracy = round_accuracy / count
    abot_accu = float(abot_accu) / count
    abot_rel_accu = float(abot_rel_accu) / count / 2
    abot_loss = float(abot_loss) / count

    if trainMode == 'fine-qbot':
        predict_loss /= numRounds
        qcycle_loss /= numRounds
        accuracy /= numRounds
    if do_perplexity:
        ques_pplx = float(ques_pplx) / (count * numRounds)
    if do_relevance:
        max_rel_prob = float(max_rel_prob) / (count * numRounds)
    if do_diversity:
        ngram_metrics = utils.distinct_ngrams(all_ques, ns=[1, 2, 3, 4])

    # print results
    result_str = (f'##### Evaluation result [accuracy: {accuracy:.4g}]'
          f'[PreictLoss: {predict_loss:.5g}]'
          f'[encoder_NLL: {encoder_nll:.5g}]'
          f'[question_matches: {question_matches:.5g}]'
          f'[question_match_ratio: {question_match_ratio:.5g}]'
          f'[cc_NLL: {cc_nll:.5g}]'
          f'[qcycle_loss: {qcycle_loss:.5g}]'
          f'[abot_accu: {abot_accu:.4g}]'
          f'[abot_rel_accu: {abot_rel_accu:.4g}]'
          f'[abot_loss: {abot_loss:.4g}]')
    if do_perplexity:
        result_str += f'[ques_pplx: {ques_pplx:.4g}]'
    if do_relevance:
        result_str += f'[max_rel_prob: {max_rel_prob:.4g}]'
    if do_diversity:
        for n in ngram_metrics:
            result_str += f'[uniq_ngram n={n}: {ngram_metrics[n]:.4g}]'
    print(result_str)
    print('round accuracies\t',end='')
    for i, racc in enumerate(list(round_accuracy)):
        print(f'R{i} {racc:.2g}  ', end='')
    print('')

    # save results to return
    evalResult = {}
    evalResult['encoder_nll'] = encoder_nll
    evalResult['question_matches'] = question_matches
    evalResult['question_match_ratio'] = question_match_ratio
    evalResult['cc_nll'] = cc_nll
    evalResult['qcycle_loss'] = qcycle_loss
    evalResult['predict_loss'] = predict_loss
    evalResult['accuracy'] = accuracy
    evalResult['abot_accu'] = abot_accu
    evalResult['abot_rel_accu'] = abot_rel_accu
    evalResult['abot_loss'] = abot_loss
    for i in range(numRounds):
        evalResult['R'+ str(i)] = round_accuracy[i]
    if do_perplexity:
        evalResult['ques_pplx'] = ques_pplx
    if do_relevance:
        evalResult['max_rel_prob'] = max_rel_prob
    if do_diversity:
        evalResult['distinct_ngrams'] = ngram_metrics

    dataset.split = original_split
    torch.set_grad_enabled(True)
    if qBot:
        qBot.tracking = initial_tracking

    return evalResult

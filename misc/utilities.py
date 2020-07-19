import os
import os.path as pth
import math
import tempfile
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from six import iteritems
from tensorboardX import SummaryWriter
import numpy as np
import pdb
import logging
import json

logger = logging.getLogger(__name__)

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def gumbel_temp_anneal_function(params, step=0, training=False):
    tau0 = params['temp']
    rate = params['tempAnnealRate']
    tmin = params['tempMin']
    hard = True
    softStart = params.get('softStartIters', 0)

    if not training:
        return float(tmin), hard
    # following https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    step = np.maximum(((step - 1) // 1000) * 1000 + 1, 0)
    temp = np.maximum(tau0 * np.exp(-rate * step), tmin)
    hard = False if step < softStart else hard
    return float(temp), hard


def raw_ques_to_str(ind2word, sens, sens_len):
    # this is a version of idx_to_str that doesn't require image or round info
    #sens = sens[:,1:]
    #sens_len = sens_len - 1
    for i in range(sens.size(0)):
        string = ''
        for j in range(sens_len[i].item()):
            string += ind2word[sens[i][j].item()]
            string += ' '
    return string


def ques_to_str(ind2word, sens, sens_len):
    # this is a version of idx_to_str that doesn't require image or round info
    sens = sens[:,1:]
    sens_len = sens_len - 1
    for i in range(sens.size(0)):
        string = ''
        for j in range(sens_len[i].item()):
            string += ind2word[sens[i][j].item()]
            string += ' '
    return string


def idx_to_str(ind2word, sens, sens_len, imgs_ids, round, decode_imgid2ques):

    sens = sens[:,1:]
    sens_len = sens_len - 1

    for i in range(sens.size(0)):
        img_id = imgs_ids[i].item()
        string = ''
        for j in range(sens_len[i].item()):
            string += ind2word[sens[i][j].item()]
            string += ' '

        if img_id not in decode_imgid2ques:
            decode_imgid2ques[img_id] = []
        decode_imgid2ques[img_id].append([string, round])

    return decode_imgid2ques


def old_idx_to_str(ind2word, sens, sens_len, imgs_ids, round, decode_question_list):

    for i in range(sens.size(0)):

        img_id = [imgs_ids[i][j].item() for j in range(len(imgs_ids[i]))]

        string = ''
        for j in range(sens_len[i].item()):
            string += ind2word[sens[i][j].item()]
            string += ' '

        decode_question_list.append([string, round, img_id])

    return decode_question_list


class TBlogger():
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + '/' + exp_name
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel='None'):
        self.logger.add_scalar(split + '/' + key, val , step)


def saveModel(model, optimizer, saveFile, params):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'params': params,
        'model_params': model.params if hasattr(model, 'params') else None,
    }, saveFile)


# use torch 1.1.0's load_state_dict for loading the speaker
_IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

def load_state_dict(self, state_dict, strict=True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.
    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(self)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)


def loadModelFromFile(fname, agent='qbot', gpu=True, speakerFile=None, verb=True):
    if verb:
        print('Loading {} from {}'.format(agent, fname))
    mdict = torch.load(fname, map_location='cpu')
    model_params = mdict['model_params']
    if speakerFile is not None:
        assert 'speakerParams' not in model_params, 'Unclear what to do here.'
    model = loadModel(model_params, agent=agent, gpu=gpu, speakerFile=speakerFile)
    missing, unexpected = load_state_dict(model, mdict['model'], strict=False)
    assert len(unexpected) == 0
    for k in missing:
        assert k.startswith('speaker.')
    model.eval()
    return model


def loadModel(model_params, agent='qbot', gpu=True, speakerFile=None):
    if agent == 'abot':
        from models.answerer import Answerer
        model = Answerer(model_params)
    elif agent == 'lm':
        from models.decoder import RNNDecoder
        model = RNNDecoder(model_params)
    elif agent == 'qbot':
        speaker = None
        if speakerFile is not None:
            print('(speaker) ', end='')
            speaker = loadModelFromFile(speakerFile, 'qbot')
            model_params['speakerParams'] = speaker.params
        from models.questioner import Questioner
        model = Questioner(model_params)
        if speaker is not None:
            model.speaker.load_state_dict(speaker.state_dict())
    else:
        raise Exception("Unknown agent {}.".format(agent))
    if gpu:
        model.cuda()
    return model


def clampGrad(grad, limit=5.0):
    '''
    Gradient clip by value
    '''
    grad.data.clamp_(min=-limit, max=limit)
    return grad


def getSortedOrder(lens):
    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().view(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    if isinstance(sortedLen, Variable):
        sortedLen = sortedLen.data
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss
    
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores    
    
def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               initialState=None,
               returnStates=False):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)

    if initialState is not None:
        hx = initialState
        sortedHx = [x.index_select(dim=1, index=fwdOrder) for x in hx]
        assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None
    _, (h_n, c_n) = rnnModel(packedSeqInput, hx)

    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        c_n = c_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output


def maskedNll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)

    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb

        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    # Shifting gtSeq 1 token left to remove <START>
    padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
    padColumn = Variable(padColumn)
    target = torch.cat([gtSeq, padColumn], dim=1)[:, 1:]

    # Generate a mask of non-padding (non-zero) tokens
    mask = target.data.gt(0)
    loss = 0
    if isinstance(gtSeq, Variable):
        mask = Variable(mask)
    assert isinstance(target, Variable)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    # Mean sentence probs:
    # gtLogProbs = gtLogProbs/(mask.float().sum(1).view(-1,1))
    if returnScores:
        return (gtLogProbs * (mask.float())).sum(1)
    maskedLL = torch.masked_select(gtLogProbs, mask)
    nll_loss = -torch.sum(maskedLL) / seq.size(0)
    return nll_loss


def concatPaddedSequences(seq1, seqLens1, seq2, seqLens2, padding='right'):
    '''
    Concates two input sequences of shape (batchSize, seqLength). The
    corresponding lengths tensor is of shape (batchSize). Padding sense
    of input sequences needs to be specified as 'right' or 'left'

    Args:
        seq1, seqLens1 : First sequence tokens and length
        seq2, seqLens2 : Second sequence tokens and length
        padding        : Padding sense of input sequences - either
                         'right' or 'left'
    '''

    concat_list = []
    cat_seq = torch.cat([seq1, seq2], dim=1)
    maxLen1 = seq1.size(1)
    maxLen2 = seq2.size(1)
    maxCatLen = cat_seq.size(1)
    batchSize = seq1.size(0)
    for b_idx in range(batchSize):
        len_1 = seqLens1[b_idx].item()
        len_2 = seqLens2[b_idx].item()

        cat_len_ = len_1 + len_2
        if cat_len_ == 0:
            raise RuntimeError("Both input sequences are empty")

        elif padding == 'left':
            pad_len_1 = maxLen1 - len_1
            pad_len_2 = maxLen2 - len_2
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][pad_len_2:]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][pad_len_1:]

            else:
                cat_ = torch.cat([seq1[b_idx][pad_len_1:],
                                  seq2[b_idx][pad_len_2:]], 0)
            cat_padded = F.pad(
                input=cat_,  # Left pad
                pad=((maxCatLen - cat_len_), 0),
                mode="constant",
                value=0)
        elif padding == 'right':
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][:len_1]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][:len_1]

            else:
                cat_ = torch.cat([seq1[b_idx][:len_1],
                                  seq2[b_idx][:len_2]], 0)
                # cat_ = cat_seq[b_idx].masked_select(cat_seq[b_idx].ne(0))
            cat_padded = F.pad(
                input=cat_,  # Right pad
                pad=(0, (maxCatLen - cat_len_)),
                mode="constant",
                value=0)
        else:
            raise (ValueError, "Expected padding to be either 'left' or \
                                'right', got '%s' instead." % padding)
        concat_list.append(cat_padded.unsqueeze(0))
    concat_output = torch.cat(concat_list, 0)
    return concat_output


#@torch._jit_internal.weak_script
def _sample_gumbel(shape, eps=1e-10, out=None):
    # type: (List[int], float, Optional[Tensor]) -> Tensor
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    if out is None:
        U = torch.rand(shape)
    else:
        U = torch.jit._unwrap_optional(out).resize_(shape).uniform_()
    return - torch.log(eps - torch.log(U + eps))


#@torch._jit_internal.weak_script
def _gumbel_softmax_sample(logits, tau=1, eps=1e-10, argmax=False):
    # type: (Tensor, float, float) -> Tensor
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    if argmax:
        y = logits #+ 0
    else:
        gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=torch.empty_like(logits))
        y = logits + gumbel_noise
    logy = F.log_softmax(y / tau, dims - 1)
    y = F.softmax(y / tau, dims - 1)
    return y, logy


#@torch._jit_internal.weak_script
def gumbel_softmax(logits, tau=1., hard=False, eps=1e-10, ret_soft=False, argmax=False):
    # type: (Tensor, float, bool, float) -> Tensor
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features

    Constraints:

    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft, logy_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps, argmax=argmax)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    if ret_soft:
        return y, logy_soft
    else:
        return y


def ngrams(sequence, n):
    # from nltk.util: https://www.nltk.org/_modules/nltk/util.html
    # TODO: is this ok to release?
    sequence = iter(sequence)
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_ngrams(sents, ns=[1, 2]):
    '''
    Return the number of unique n-grams in a list of sentences, normalizing
    by the total number of words.
    '''
    sents = [sent.split(' ') for sent in sents]
    metrics = {}
    for n in ns:
        unique_ngrams, n_total_ngrams = _compute_distinct_ngrams(sents, n)
        metrics[n] = len(unique_ngrams) / n_total_ngrams
    return metrics


def _compute_distinct_ngrams(sents, n):
    # compute the count of distinct ngrams in all sents (not normalized)
    # TODO: what about...
    # <UNK>, ', ?
    all_ngrams = []
    for sent in sents:
        all_ngrams.extend(iter(ngrams(sent, n)))
    n_total_ngrams = len(all_ngrams)
    unique_ngrams = list(set(all_ngrams))
    return unique_ngrams, n_total_ngrams

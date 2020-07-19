import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from misc import utilities as utils
import pdb

infty = float('inf')

class RNNDecoder(nn.Module):
    def __init__(self, params):
        super(RNNDecoder, self).__init__()
        self.params = params
        self.vocabSize = params['vocabSize']
        self.embedSize = params['embedSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.numLayers = params['numLayers']
        self.startToken = params['startToken']
        self.endToken = params['endToken']
        self.dropout = params['dropout']
        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.decoderVar = params['decoderVar']
        # Modules
        self.rnn = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.wordEmbed = nn.Sequential(
                nn.Embedding(self.vocabSize, self.embedSize, padding_idx=0),
                nn.ReLU(),
                nn.Dropout(0.3))

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = next(self.parameters()).data
        h = someTensor.new(self.numLayers, self.batchSize, self.rnnHiddenSize).zero_()
        c = someTensor.new(self.numLayers, self.batchSize, self.rnnHiddenSize).zero_()
        return (h, c)


    # TODO: remove imgs parameter
    def forward(self, encStates, imgs, inputSeq):
        '''
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.

        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities

        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at evert time step

        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        '''
        self.batchSize = inputSeq.shape[0]
        hidden = self._initHidden()

        if encStates is not None:
            _, hidden = self.rnn(encStates.unsqueeze(1), hidden)

        if inputSeq is not None:
            inputSeq = self.wordEmbed(inputSeq)
            outputs, _ = self.rnn(inputSeq, hidden)
            outputs = F.dropout(outputs.contiguous(), self.dropout, training=self.training)
            outputSize = outputs.size()
            flatOutputs = outputs.view(-1, outputSize[2])
            flatScores = self.outNet(flatOutputs)
            if self.decoderVar == 'cat':
                flatLogProbs = self.logSoftmax(flatScores)
            if self.decoderVar == 'gumbel':
                _, flatLogProbs = utils.gumbel_softmax(flatScores, ret_soft=True)
            logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)
        return logProbs

    # TODO: remove imgs parameter
    def forwardDecode(self,
                      encStates,
                      imgs,
                      maxSeqLen=20,
                      inference='sample',
                      beamSize=1):
        '''
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width

        Notes:
            * This function is not called during SL pre-training
            * Greedy inference is used for evaluation
            * Sampling is used in RL fine-tuning
        '''
        if inference == 'greedy' and beamSize > 1:
            # Use beam search inference when beam size is > 1
            return self.beamSearchDecoder(encStates, beamSize, maxSeqLen)

        # Determine if cuda tensors are being used
        if self.outNet.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        self.samples = []
        maxLen = maxSeqLen + 1  # Extra <END> token
        self.batchSize = batchSize = encStates.size(0)
        # Placeholder for filling in tokens at evert time step
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, 0] = self.startToken
        seq = Variable(seq, requires_grad=False)
        if self.decoderVar == 'gumbel':
            oneHotSeq = th.FloatTensor(batchSize, maxLen, self.vocabSize)
            oneHotSeq.fill_(0)
            oneHotSeq[:, 0, self.startToken] = 1.0
        else:
            oneHotSeq = None

        # Initial state linked from encStates
        hid = self._initHidden()
        # TODO: de-duplicate from forward()?
        _, hid = self.rnn(encStates.unsqueeze(1), hid)

        sampleLens = th.LongTensor(batchSize).fill_(0)
        # Tensors needed for tracking sampleLens
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.BoolTensor(seq.size()).fill_(0)

        self.saved_log_probs = []

        # Generating tokens sequentially
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            # emb has shape  (batch, 1, embedSize)
            output, hid = self.rnn(emb, hid)
            # output has shape (batch, 1, rnnHiddenSize)
            scores = self.outNet(output.squeeze(1))

            # Explicitly removing padding token (index 0) and <START> token
            # (index -2) from scores so that they are never sampled.
            # This is allows us to keep <START> and padding token in
            # the decoder vocab without any problems in RL sampling.
            if t > 0:
                scores[:, 0] = -infty
                scores[:, -2] = -infty
            elif t == 0:
                # Additionally, remove <END> token from the first sample
                # to prevent the sampling of an empty sequence.
                scores[:, 0] = -infty
                scores[:, -2:] = -infty

            if inference == 'sample' and self.decoderVar == 'cat':
                logProb = self.logSoftmax(scores)
                probs = torch.exp(logProb)
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                # Saving log probs for a subsequent reinforce call
                self.saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)
            elif inference == 'sample' and self.decoderVar == 'gumbel':
                oneHotSeq[:, t+1, :] = utils.gumbel_softmax(scores, hard=True)
                sample = oneHotSeq[:, t+1, :].max(dim=1)[1]
                sample = sample.unsqueeze(-1)
            elif inference == 'greedy' and self.decoderVar == 'gumbel':
                oneHotSeq[:, t+1, :] = utils.gumbel_softmax(scores, hard=True, argmax=True)
                sample = oneHotSeq[:, t+1, :].max(dim=1)[1]
                sample = sample.unsqueeze(-1)
            elif inference == 'greedy':
                logProb = self.logSoftmax(scores)
                _, sample = torch.max(logProb, dim=1, keepdim=True)
            else:
                raise ValueError(
                    "Invalid inference type: '{}'".format(inference))

            self.samples.append(sample)

            seq.data[:, t + 1] = sample.data.squeeze(1)
            # Marking spots where <END> token is generated
            mask[:, t] = sample.data.eq(self.endToken).squeeze(1)

        mask[:, maxLen - 1].fill_(1)

        # Computing lengths of generated sequences
        for t in range(maxLen):
            # Zero out the spots where end token is reached
            unitColumn.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unitColumn
            # Add +1 length to all un-ended sequences
            sampleLens = sampleLens + unitColumn

        # Keep mask for later use in RL reward masking
        self.mask = Variable(mask, requires_grad=False)

        # Adding <START> and <END> to generated answer lengths for consistency
        maxLenCol = torch.zeros_like(sampleLens) + maxLen
        sampleLens = torch.min(sampleLens + 1, maxLenCol)
        sampleLens = Variable(sampleLens, requires_grad=False)

        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)

        # Note that we do not add startColumn to self.samples itself
        # as reinforce is called on self.samples (which needs to be
        # the output of a stochastic function)
        gen_samples = [startColumn] + self.samples

        samples = torch.cat(gen_samples, 1)
        result = {
            'samples': samples,
            'sampleLens': sampleLens,
            'sampleOneHot': oneHotSeq,
        }
        return result

    def evalOptions(self, encStates, options, optionLens, scoringFunction):
        '''
        Forward pass a set of candidate options to get log probabilities

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            options   : (batchSize, numOptions, maxSequenceLength) sized
                        tensor with <START> and <END> tokens

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.

        Output:
            A (batchSize, numOptions) tensor containing the score
            of each option sentence given by the generator
        '''
        batchSize, numOptions, maxLen = options.size()
        optionsFlat = options.contiguous().view(-1, maxLen)

        # Reshaping H, C for each option
        encStates = [x.unsqueeze(2).repeat(1,1,numOptions,1).\
                        view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in encStates]

        logProbs = self.forward(encStates, inputSeq=optionsFlat)
        scores = scoringFunction(logProbs, optionsFlat, returnScores=True)
        return scores.view(batchSize, numOptions)

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen):
        '''
        Beam search for sequence generation

        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        '''
        raise Exception('No longer supported. Needs update for gumbel var.')

        # For now, use beam search for evaluation only
        assert self.training == False

        # Determine if cuda tensors are being used
        if self.outNet.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True
        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = initStates.size(0)

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)

        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange)

        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = self._initHidden()

        _, hiddenStates = self.rnn(initStates.unsqueeze(1), hiddenStates)

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
            self.endToken)
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)

        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

        for t in range(maxLen - 1):  # Beam expansion till maxLen]
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                # emb has shape (batchSize, 1, embedSize)

                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores)
                # scores & logProbs has shape (batchSize, vocabSize)

                # Find top beamSize logProbs
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                # pdb.set_trace()
                beamTokensTable[:, :, 0] = topIdx.data
                logProbSums = topLogProbs

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]
                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(
                    emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                # logProbs has shape (batchSize*beamSize, vocabSize)
                # NOTE: Padding token has been removed from generator output during
                # sampling (RL fine-tuning). However, the padding token is still
                # present in the generator vocab and needs to be handled in this
                # beam search function. This will be supported in a future release.
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                       self.vocabSize)
                
                if LENGTH_NORM:
                    # Add (current log probs / (t+1))
                    logProbs = logProbsCurrent * (aliveVector.float() /
                                                  (t + 1))
                    # Add (previous log probs * (t/t+1) ) <- Mean update
                    coeff_ = aliveVector.eq(0).float() + (
                        aliveVector.float() * t / (t + 1))
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    # Add currrent token logProbs for alive beams only
                    logProbs = logProbsCurrent * (aliveVector.float())
                    # Add previous logProbSums upto t-1
                    logProbs += logProbSums.unsqueeze(2)

                # Masking out along |V| dimension those sequence logProbs
                # which correspond to ended beams so as to only compare
                # one copy when sorting logProbs
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :,
                      0] = 0  # Zeroing all except first row for ended beams
                minus_infinity_ = torch.min(logProbs).item()
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                repeat(batchSize,beamSize,1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).\
                                repeat(1,1,self.vocabSize).view(batchSize,-1)

                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                logProbSums = topLogProbs
                beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Update corresponding hidden and cell states for next time step
                hiddenCurrent, cellCurrent = hiddenStates

                # Reshape to get explicit beamSize dim
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)

                # Update states according to the next top beams
                backIndexVector = backIndices[:, :, t].unsqueeze(0)\
                    .unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)

                # Restore original shape for next rnn forward
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = (hiddenCurrent, cellCurrent)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        RECOVER_TOP_BEAM_ONLY = True

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:,:,tokenIdx].\
                        gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].\
                        gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens = tokens[:, 0]  # Keep only top beam
            seqLens = seqLens[:, 0]

        return Variable(tokens), Variable(seqLens)

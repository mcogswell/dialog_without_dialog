import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utilities as utils
import pdb
import numpy as np
from models.context_coder import ImageSetContextCoder
from models.answerer import FCNet

class RNNEncoder(nn.Module):
    def __init__(self, params, useImage=False):
        super(RNNEncoder, self).__init__()

        self.rnnHiddenSize = params['rnnHiddenSize']
        self.numLayers = params['numLayers']
        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.dropout = params['dropout']
        self.startToken = params['startToken']
        self.endToken = params['endToken']
        self.dropout = params['dropout']
        self.vocabSize = params['vocabSize']
        self.embedSize = params['embedSize']
        self.useImage = useImage

        if self.useImage:
            self.ImageSetContextCoder = ImageSetContextCoder(
                                            self.imgEmbedSize,
                                            self.imgFeatureSize,
                                            self.rnnHiddenSize)
            self.f_net = FCNet([self.rnnHiddenSize + self.imgEmbedSize,
                                self.rnnHiddenSize])

        self.quesRNN = nn.LSTM(self.embedSize, self.rnnHiddenSize,
                                self.numLayers, batch_first=True)    
        self.wordEmbed = nn.Sequential(
                nn.Embedding(self.vocabSize, self.embedSize, padding_idx=0),
                nn.ReLU(),
                nn.Dropout(0.3))

        # then the output of the GRU network and image feature together do the self attention
        # over the image pool
        # we can have the CVAE augment version which predict the latent vsariable.

    def processSequence(self, seq, seqLen, seqOneHot):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        if seqOneHot is not None:
            seqOneHot = seqOneHot[:, 1:]
        return seq[:, 1:], seqLen - 1, seqOneHot

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = next(self.parameters()).data
        h = someTensor.new(self.numLayers, self.batchSize, self.rnnHiddenSize).zero_()
        c = someTensor.new(self.numLayers, self.batchSize, self.rnnHiddenSize).zero_()
        return (h, c)

    def forward(self, ques, ques_len, imgs=None, ques_one_hot=None):

        batchSize = ques.size(0)
        ques, ques_len, ques_one_hot = self.processSequence(ques.clone(), ques_len.clone(), ques_one_hot)
        if ques_one_hot is None:
            quesIn = self.wordEmbed(ques)
        else:
            # this allows gradients to flow for gumbel softmax
            quesIn = torch.matmul(ques_one_hot, self.wordEmbed[0].weight)
            quesIn = self.wordEmbed[1:](quesIn)

        if self.useImage:
            oupt = utils.dynamicRNN(self.quesRNN, quesIn, ques_len)
            iEmbed = self.ImageSetContextCoder(imgs, oupt)
            oupt = self.f_net(torch.cat([iEmbed, oupt], dim=1))
        else:
            oupt = utils.dynamicRNN(self.quesRNN, quesIn, ques_len)

        return oupt

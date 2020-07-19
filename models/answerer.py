import torch
import torch.nn as nn
import torch.nn.functional as F
from models.agent import Agent
from misc import utilities as utils
import pdb
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        
        # import pdb
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class Answerer(Agent):
    def __init__(self, params):
        super(Answerer, self).__init__()

        self.params = params.copy()
        self.ansObjective = params['ansObjective']
        self.latentSize = params['latentSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.wordDropoutRate = params['wordDropoutRate']
        self.unkToken = params['unkToken']
        self.endToken = params['endToken']
        self.startToken = params['startToken']
        self.embedSize = params['embedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.num_ans_candidates = params['num_ans_candidates']
        self.vocabSize = params['vocabSize']
        self.numLayers = params['numLayers']

        self.w_emb = nn.Embedding(self.vocabSize, self.embedSize, padding_idx=0)
        self.quesRNN = nn.LSTM(self.embedSize, self.rnnHiddenSize, self.numLayers, batch_first=True)

        self.v_att = NewAttention(self.imgFeatureSize, self.rnnHiddenSize, 1024)
        self.q_net = FCNet([self.rnnHiddenSize, 1024])
        self.v_net = FCNet([self.imgFeatureSize, 1024])
        self.classifier = SimpleClassifier(
            1024, 1024 * 2, self.num_ans_candidates, 0.5)

        if self.ansObjective == 'both':
            self.relavency_classifier = SimpleClassifier(
            1024, 1024 * 2, 2, 0.5)

        self.tracking = params.get('abotTrackAttention', False)

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def forward(self, v, q, q_len, rand_q=None, rand_q_len=None, inference_mode=False):
        # assert inference_mode in [False, 'norel']

        v = v.squeeze(1)
        q, q_len = self.processSequence(q.clone(), q_len.clone())
        w_emb = self.w_emb(q)
        q_emb = utils.dynamicRNN(self.quesRNN, w_emb, q_len)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        if self.tracking:
            self.v_emb = v_emb

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        relavent_logit_1 = None
        relavent_logit_2 = None

        if self.ansObjective == 'both':
            relavent_logit_1 = self.relavency_classifier(joint_repr)
            if not inference_mode in [False, 'norel']:
                rand_q, rand_q_len = self.processSequence(rand_q.clone(), rand_q_len.clone())
                rand_w_emb = self.w_emb(rand_q)
                rand_q_emb = utils.dynamicRNN(self.quesRNN, rand_w_emb, rand_q_len)

                rand_att = self.v_att(v, rand_q_emb)
                rand_v_emb = (rand_att * v).sum(1) # [batch, v_dim]

                rand_q_repr = self.q_net(rand_q_emb)
                rand_v_repr = self.v_net(rand_v_emb)
                rand_joint_repr = rand_q_repr * rand_v_repr
                relavent_logit_2 = self.relavency_classifier(rand_joint_repr)

        if inference_mode == 'norel':
            return logits
        elif inference_mode == False:
            return logits, relavent_logit_1
        else:
            return logits, relavent_logit_1, relavent_logit_2

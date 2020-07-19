import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from models.agent import Agent
import models.encoder as enc
import models.decoder as dec
from models.answerer import FCNet, SimpleClassifier
import models.context_coder as ctx
from misc.vector_quantizer import VectorQuantizerEMA, VectorQuantizer
from misc import utilities as utils
from misc.utilities import gumbel_softmax
import pdb


class Questioner(Agent):
    def __init__(self, params):
        '''
        
        '''
        super(Questioner, self).__init__()

        self.params = params.copy()
        self.varType = params['varType']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.latentSize = params['latentSize']
        self.wordDropoutRate = params['wordDropoutRate']
        self.unkToken = params['unkToken']
        self.endToken = params['endToken']
        self.startToken = params['startToken']
        self.embedSize = params['embedSize']
        self.num_embeddings = params['num_embeddings']
        self.num_vars = params.get('num_vars', 1)
        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.ansEmbedSize = params['ansEmbedSize']
        self.num_ans_candidates = params['num_ans_candidates']
        self.numLayers = params['numLayers']
        self.query_type = params.get('queryType', 'dialog_only')
        self.speaker_type = params.get('speakerType', 'two_part')

        self._num_embeddings = self.num_embeddings * self.num_vars
        # by default, use the final annealed temperature
        temp, hard = utils.gumbel_temp_anneal_function(params, training=False)
        self.temp = temp
        self.hard = hard

        # c context (images)
        # x question
        # z latent variable
        # a answer
        # y index in image pool

        # Encoder q(z | x, c)
        # x, c -> h_enc
        self.encoder = enc.RNNEncoder(params)
        # h_enc -> z
        if self.varType == 'cont':
            self.henc2z = nn.ModuleList([
                    nn.Linear(self.rnnHiddenSize, self.latentSize), # mean
                    nn.Linear(self.rnnHiddenSize, self.latentSize), # logv
                ])
        elif self.varType == 'gumbel':
            self.henc2z = nn.Linear(self.rnnHiddenSize, self._num_embeddings)
        elif self.varType == 'none':
            self.henc2z = lambda x: x

        # Policy / Context Coder p(z | c)
        # initialize this even for cond_vae because it will be fine-tuned late
        # c -> h_c
        self.ctx_coder = ctx.ImageSetContextCoder(self.imgEmbedSize,
                                                  self.imgFeatureSize,
                                                  self.rnnHiddenSize)
        # h_c -> z
        if self.varType == 'cont':
            self.hc2z = nn.ModuleList([
                    nn.Linear(self.imgEmbedSize, self.latentSize), # mean
                    nn.Linear(self.imgEmbedSize, self.latentSize), # logv
                ])
            self.z_dim = self.latentSize
        elif self.varType == 'gumbel':
            self.hc2z = nn.Linear(self.imgEmbedSize, self._num_embeddings)
            self.z_dim = self._num_embeddings
        elif self.varType == 'none':
            self.hc2z = lambda x: x
            self.z_dim = self.imgEmbedSize

        # we may need to tune this.
        self.dialogRNN = dialogRNN(params)
        if self.speaker_type == 'two_part_zgrad':
            self.z2dialog = nn.Sequential(
                    nn.Linear(self.z_dim, self.rnnHiddenSize),
                    nn.ReLU(),
                )
        elif self.speaker_type == 'two_part_zgrad_logit':
            self.zlogit2dialog = nn.Sequential(
                    nn.Linear(self.z_dim, self.rnnHiddenSize),
                    nn.ReLU(),
                )
        elif self.speaker_type == 'two_part_zgrad_codebook':
            self.codebook2dialog = nn.Sequential(
                    nn.Linear(self.embedSize, self.rnnHiddenSize),
                    nn.ReLU(),
                )

        # answer embedding from last round for the latent space. The begining is a start token. 
        self.ans_uncertain_token = self.num_ans_candidates + 0
        self.ans_start_token = self.num_ans_candidates + 1
        self.ans_not_relevant_token = self.num_ans_candidates + 2
        # TODO: fix typo, but don't break compatibility with existing trained models
        self.ansEmebed = nn.Embedding(self.num_ans_candidates+3, self.ansEmbedSize)

        self.quesStartEmbed = nn.Embedding(1, self.rnnHiddenSize)
        self.quesEncoder = enc.RNNEncoder(params, useImage=False)

        # Decoder p(x | z)
        # z -> h_dec
        if self.varType == 'gumbel':
            new_codebook = lambda: nn.Linear(self.num_embeddings, self.latentSize, bias=False)
            codebooks = [new_codebook() for _ in range(self.num_vars)]
            self.gumbel_codebook = nn.ModuleList(codebooks)
        self.z2hdec = nn.Linear(self.latentSize, self.embedSize)
        # h_dec -> x
        self.decoder = dec.RNNDecoder(params)

        # Predict image p(y)
        self.predict_q = FCNet([self.rnnHiddenSize*2+self.ansEmbedSize, 1024])
        self.predict_ctx = ctx.ImageContextCoder(params)
        self.predict_fact = ctx.FactContextCoder(params)
        self.predict_logit = SimpleClassifier(1024, 1024*2, 1, 0.5)

        # when true, save various internal state (e.g., z)
        self.tracking = False

        # The speaker is another qbot (usually a pre-trained model kept fixed)
        # which can be used to generate questions to get a model that can't
        # change its dialog state.
        if self.params.get('speakerParams', None) is not None:
            self.speaker = utils.loadModel(params['speakerParams'], 'qbot')
            self.speaker.tracking = True
            # when speaker mode is on, also have the speaker track the dialog
            self.speakerMode = True
        else:
            self.speakerMode = False

        if self.query_type == 'dialog_qa':
            # dialogRNN embed + ans embed + ques embed
            query_embed_in_dim = self.rnnHiddenSize + self.ansEmbedSize + self.rnnHiddenSize
            self.query_embed = nn.Linear(query_embed_in_dim, self.rnnHiddenSize)

    @property
    def tracking(self):
        return self._tracking

    @tracking.setter
    def tracking(self, value):
        self._tracking = value
        self.ctx_coder.tracking = value
        self.predict_ctx.tracking = value
        if hasattr(self, 'dis_ctx'):
            self.dis_ctx.tracking = value
        if (hasattr(self, 'encoder') and 
            hasattr(self.encoder, 'ImageSetContextCoder')):
            self.encoder.ImageSetContextCoder.tracking = value

    def _h2z(self, h2z, h, inference='sample'):
        assert inference in ['sample', 'greedy']
        batch_size = h.size(0)
        if self.varType == 'cont':
            # REPARAMETERIZATION
            mean = h2z[0](h)
            logv = h2z[1](h)
            std = torch.exp(0.5 * logv)
            if inference == 'sample':
                sample = torch.randn([batch_size, self.latentSize]).type_as(h)
                z = sample * std + mean
            elif inference == 'greedy':
                z = mean
            return z, (mean, logv)
        elif self.varType == 'gumbel':
            # TODO: use this as history representation
            z_logit_param = h2z(h)
            temp = self.temp
            hard = self.hard
            K = self.num_embeddings
            V = self.num_vars
            z_v_logits = [z_logit_param[:, v*K:(v+1)*K] for v in range(V)]
            z_v_logprobs = [F.log_softmax(zv_logit, dim=1) for zv_logit in z_v_logits]
            z = None
            z_soft = None
            # compute z
            if inference == 'sample':
                z_vs = [gumbel_softmax(z_vl, tau=temp, hard=hard, ret_soft=True) for z_vl in z_v_logprobs]
                z = torch.cat([z[0] for z in z_vs], dim=1)
                z_soft = torch.cat([z[1] for z in z_vs], dim=1)
            # TODO: is this really the argmax of the gumbel softmax?
            elif inference == 'greedy' and not hard:
                z_vs = [F.softmax(z_vl / temp, dim=1) for z_vl in z_v_logprobs]
                z = torch.cat([z for z in z_vs], dim=1)
            elif inference == 'greedy' and hard:
                idxs = [z_vl.max(dim=1, keepdim=True)[1] for z_vl in z_v_logprobs]
                z_vs = [torch.zeros_like(z_vl).scatter_(1, idx, 1.0) for z_vl, idx in zip(z_v_logprobs, idxs)]
                z = torch.cat([z for z in z_vs], dim=1)
            z_logprob = torch.cat(z_v_logprobs, dim=1)
            return z, (z_logprob, z_soft)
        elif self.varType == 'none':
            z = h
            return z, None

    def reset(self):
        self.dialogHiddens = []
        self.questions = []
        self.quesOneHot = []
        self.quesLens = []

        self.gt_questions = []
        self.gt_quesLens = []

        self.rand_questions = []
        self.rand_quesLens = []

        self.answers = []
        self.dialogQuerys = []
        self.images = None
        self.batch_size = 0
        self.latent_states = []

        self.dialogQuesEmbedding = []
        self.dialogAnsEmbedding = []
        if self.speakerMode:
            self.speaker.reset()

    def observe(self, *args, **kwargs):
        self._observe(*args, **kwargs)
        if self.speakerMode:
            self.speaker.observe(*args, **kwargs)

    def _observe(self, images=None,
                 ques=None, ques_len=None, ques_one_hot=None, gt_ques=False,
                 ans=None, ans_rel=None,
                 start_answer=False, start_question=False):
        
        if images is not None:
            self.images = images
            self.batch_size = images.size(0)
        
        if ques is not None:
            if gt_ques:
                self.gt_questions.append(ques)
            else:
                self.questions.append(ques)

        if ques_len is not None:
            if gt_ques:
                self.gt_quesLens.append(ques_len)
            else:
                self.quesLens.append(ques_len)

        if ques_one_hot is not None:
            assert not gt_ques
            self.quesOneHot.append(ques_one_hot)

        if ans is not None:
            if ans_rel is not None:
                ans[ans_rel==0] = self.ans_not_relevant_token
                
            self.answers.append(ans)

        if start_answer:
            self.answers.append(torch.full([self.batch_size], self.ans_start_token, dtype=torch.long, device=self.images.device))

        if start_question:
            self.questions.append(torch.full([self.batch_size], 0, dtype=torch.long, device=self.images.device))
            self.quesLens.append(-1)

    def embedDialog(self, inpt, ques, ques_len, answer, ques_one_hot=None):
        if len(self.dialogHiddens) == 0:
            batchSize = ques.shape[0]
            hPrev = self._initHidden(batchSize)
            quesEmbedding = self.quesStartEmbed(ques)
        else:
            hPrev = self.dialogHiddens[-1]
            # rnn question encoder here. 
            quesEmbedding = self.quesEncoder(ques, ques_len, ques_one_hot=ques_one_hot)
        
        # embed the answer here. We didn't connect the output embedding yet. 
        ansEmbedding = self.ansEmebed(answer)
        oupt, query, hNew = self.dialogRNN(inpt, quesEmbedding, ansEmbedding, hPrev)

        # add residual connection
        oupt = oupt.squeeze(1) + inpt
        self.dialogHiddens.append(hNew)
        if self.query_type == 'dialog_only':
            self.dialogQuerys.append(query)
            self.dialogQuesEmbedding.append(quesEmbedding)
            self.dialogAnsEmbedding.append(ansEmbedding)
        elif self.query_type == 'dialog_qa':
            self.dialogQuesEmbedding.append(quesEmbedding)
            self.dialogAnsEmbedding.append(ansEmbedding)
            query = torch.cat([query, ansEmbedding, quesEmbedding], dim=1)
            query = self.query_embed(query)
            self.dialogQuerys.append(query)

        return oupt

    def _initHidden(self, batchSize):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert batchSize != 0, 'Observe something to infer batch size.'
        someTensor = next(self.parameters()).data
        h = someTensor.new(batchSize, self.rnnHiddenSize).zero_()
        c = someTensor.new(batchSize, self.rnnHiddenSize).zero_()
        return (h, c)

    def _z2hdec(self, z):
        z, _ = z
        # z -> h_dec
        if self.varType == 'cont':
            h_dec = self.z2hdec(z)
        elif self.varType == 'gumbel':
            K = self.num_embeddings
            gumbel_embed = 0
            for v in range(self.num_vars):
                z_v = z[:, v*K:(v+1)*K]
                gumbel_embed += self.gumbel_codebook[v](z_v)
            # TODO: use gumbel_embed as the history representation
            h_dec = self.z2hdec(gumbel_embed)
        elif self.varType == 'none':
            h_dec = z
        return h_dec

    def _klloss(self, z1, z2):
        # gradients only pass through z1, not z2
        if self.varType == 'cont' and z2 == 'prior':
            mean, logv = z1[1]
            kl = -0.5 * torch.mean(1 + logv - mean.pow(2) - logv.exp())
        elif self.varType == 'gumbel' and z2 == 'prior':
            z, z_param = z1
            z_logprob, _ = z_param
            K = self.num_embeddings
            # KL loss term(s)
            # with logit parameters (Eric Jang's version)
            #q_v = F.softmax(z1[1].reshape([-1, K]), dim=1)
            # with samples
            log_q_v = z_logprob.reshape([-1, K])
            q_v = log_q_v.exp()
            logprior = torch.tensor(1. / self.num_embeddings).to(q_v).log()
            kl = (q_v * (log_q_v - logprior)).sum(dim=1)
            kl = kl.mean() # over variables and batch
        elif self.varType == 'none' and z2 == 'prior':
            kl = 0
        return kl

    def _infer_z(self, method, inference='sample'):
        # we will try different auto-encoder method for question generation. 
        # first with the continous vae, with various technique to make the latent code informative.
        # batch_size = imgs.shape[0]
        # if the length of the answer is 0         

        # x, c -> z_enc
        if method == 'encoder':
            h_enc = self.encoder(self.gt_questions[-1], ques_len=self.gt_quesLens[-1], imgs=self.images)
            z = self._h2z(self.henc2z, h_enc, inference=inference)

        # c -> z_c; also advances dialog state; (prior method is for a baseline)
        if method in ['policy', 'prior', 'speaker']:
            if len(self.dialogHiddens) == 0:
                h_c = self.ctx_coder(self.images, None)
            else:
                h_c = self.ctx_coder(self.images, self.dialogQuerys[-1])
            # NOTE: h_c is not the recurrent
            # add an RNN model here and embed previous answers.
            h_c = self.embedDialog(h_c,
                                   self.questions[-1],
                                   self.quesLens[-1],
                                   self.answers[-1],
                                   self.quesOneHot[-1] if self.quesOneHot else None)

            if self.speakerMode:
                # allow evaluation of the policy in speakerMode
                assert method == 'speaker' or not self.training
            if method == 'policy':
                z = self._h2z(self.hc2z, h_c, inference=inference)
            elif method == 'prior':
                z = self._prior(inference=inference)
            elif method == 'speaker':
                with torch.no_grad():
                    self.speaker.forwardDecode(z_inference=inference,
                                               z_source='policy')
                    z = self.speaker.z
            self.latent_states.append(z)
            # allow gradients to flow to z from the dialog rnn
            if self.speaker_type == 'two_part_zgrad':
                _h, _c = self.dialogHiddens[-1]
                self.dialogHiddens[-1] = (_h + self.z2dialog(z[0]), _c)
            elif self.speaker_type == 'two_part_zgrad_logit':
                assert self.varType == 'gumbel', 'just use two_part_zgrad'
                _h, _c = self.dialogHiddens[-1]
                self.dialogHiddens[-1] = (_h + self.zlogit2dialog(z[1][0]), _c)
            elif self.speaker_type == 'two_part_zgrad_codebook':
                assert self.varType == 'gumbel', 'just use two_part_zgrad'
                _h, _c = self.dialogHiddens[-1]
                h_codebook = self._z2hdec(z)
                self.dialogHiddens[-1] = (_h + self.codebook2dialog(h_codebook), _c)
        return z

    def forward(self):
        # compute z from encoder and/or policy
        z_enc = self._infer_z('encoder')
        z_c = self._infer_z('policy')

        # compute question logprobs and regularize zs
        logProbs = self._decodez(z_enc)
        kl = self._klloss(z_enc, 'prior')
        ccLogProbs = self._decodez(z_c)
        cckl = self._klloss(z_c, 'prior')

        return logProbs, kl, ccLogProbs, cckl

    def _decodez(self, z):
        # z -> h_dec
        h_dec = self._z2hdec(z)
        # h_dec -> x
        # decoder input word dropout
        gt_ques = self.gt_questions[-1]

        if self.wordDropoutRate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(gt_ques.size()).type_as(h_dec)
            prob[(gt_ques == self.unkToken) | (gt_ques == 0) | (gt_ques == self.startToken) | (gt_ques == self.endToken)] = 1
            gt_ques = gt_ques.clone()
            gt_ques[prob < self.wordDropoutRate] = self.unkToken

        logProbs = self.decoder(h_dec, self.images, gt_ques)
        return logProbs

    def _prior(self, inference='sample'):
        device = self.z2hdec.weight.device
        batchSize = self.images.shape[0]
        if self.varType == 'cont':
            if inference == 'sample':
                sample = torch.randn(batchSize, self.latentSize, device=device)
            else:
                sample = torch.zeros(batchSize, self.latentSize, device=device)
            mean = torch.zeros_like(sample)
            logv = torch.ones_like(sample)
            z = sample, (mean, logv)
        elif self.varType == 'gumbel':
            K = self.num_embeddings
            V = self.num_vars
            prior_probs = torch.tensor([1 / K] * K,
                                       dtype=torch.float, device=device)
            logprior = torch.log(prior_probs)
            if inference == 'sample':
                prior = OneHotCategorical(prior_probs)
                z_vs = prior.sample(sample_shape=(batchSize * V,))
                z = z_vs.reshape([batchSize, -1])
            else:
                z_vs = prior_probs.expand(batchSize * V, -1)
                z = z_vs.reshape([batchSize, -1])
            z = (z, logprior)
        elif self.varType == 'none':
            raise Exception('Z has no prior for varType==none')
        return z

    def forwardDecode(self, dec_inference='sample', beamSize=1, maxSeqLen=20,
                      z_inference='sample', z_source='encoder'):
        '''
        Decode a sequence (question) using either sampling or greedy inference.
        This can be called after observing necessary context using observe().

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            z_source  : Where to get z from?
                'encoder' - Encode the question and image pool into a z (default)
                'policy' - Encode the image pool via context coder
                'prior' - Sample a z from the prior on z (without any context)
                'speaker' - Sample z from the speaker model (must exist)
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        '''
        # infer decoder initial state
        if torch.is_tensor(z_source):
            z = (z_source, None)
        elif z_source in ['policy', 'encoder', 'prior', 'speaker']:
            z = self._infer_z(z_source, inference=z_inference)
        else:
            raise Exception('Unknown z_source {}'.format(z_source))
        h_dec = self._z2hdec(z)

        # decode z
        dec_out = self.decoder.forwardDecode(
            h_dec,
            self.images,
            maxSeqLen=maxSeqLen,
            inference=dec_inference,
            beamSize=beamSize)

        if self.tracking:
            self.z = z
        return dec_out['samples'], dec_out['sampleLens'], dec_out['sampleOneHot']

    def predictImage(self):
        '''
        Given the question answer pair, and image feature. predict 
        whether the image fit for the QA pair.
        '''
        assert len(self.dialogHiddens) != 0
        # encode history and dialog recurrent state into query
        hidden = self.dialogHiddens[-1][0].squeeze(0)
        quesFact = torch.cat([embed.unsqueeze(1) for embed in self.dialogQuesEmbedding], dim=1)
        ansFact = torch.cat([embed.unsqueeze(1) for embed in self.dialogAnsEmbedding], dim=1)
        qaFact = self.predict_fact(torch.cat((quesFact, ansFact), dim=2), hidden)
        query = torch.cat([hidden, qaFact], dim=1)

        # use query to attend to pool images and further embed history
        iEmbed = self.predict_ctx(self.images, query)
        qEmbed = self.predict_q(query)

        # final logits
        logit = self.predict_logit(iEmbed * qEmbed.unsqueeze(1).expand_as(iEmbed))
        return logit.squeeze(2)


class dialogRNN(nn.Module):
    def __init__(self, params):
        super(dialogRNN, self).__init__()
        # build the memory module for the questioner.
        self.quesEmbedSize = self.rnnHiddenSize = params['rnnHiddenSize']
        self.numLayers = params['numLayers']
        self.imgEmbedSize = params['imgEmbedSize']
        self.ansEmbedSize = params['ansEmbedSize']
        self.dropout = params['dropout']
        self.speaker_type = params.get('speakerType', 'two_part')
        if self.speaker_type.startswith('two_part'):
            self.in_dim = self.ansEmbedSize + self.imgEmbedSize + self.quesEmbedSize
        elif self.speaker_type == 'one_part':
            self.in_dim = self.ansEmbedSize + self.quesEmbedSize

        self.rnn = nn.LSTMCell(self.in_dim, self.rnnHiddenSize) # we, fc, h^2_t-1
        self.i2h_1 = nn.Linear(self.in_dim, self.rnnHiddenSize)
        self.i2h_2 = nn.Linear(self.in_dim, self.rnnHiddenSize)

        self.h2h_1 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
        self.h2h_2 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)

    def forward(self, img_feat, quesEmbedding, ans_embedding, state):
        if self.speaker_type.startswith('two_part'):
            lstm_input = torch.cat([img_feat, quesEmbedding, ans_embedding], dim=1)
        elif self.speaker_type == 'one_part':
            lstm_input = torch.cat([quesEmbedding, ans_embedding], dim=1)
        ada_gate1 = torch.sigmoid(self.i2h_1(lstm_input) + self.h2h_1(state[0]))
        #ada_gate2 = torch.sigmoid(self.i2h_2(lstm_input) + self.h2h_2(state[0]))

        hidden, cell = self.rnn(lstm_input, (state[0], state[1]))

        output = F.dropout(hidden, self.dropout, self.training)
        query = F.dropout(ada_gate1*torch.tanh(cell), self.dropout, training=self.training)

        state = (hidden, cell)

        return output, query, state


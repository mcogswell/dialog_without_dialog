import torch
import torch.nn as nn
import torch.nn.functional as F

import options
import misc.utilities as utils

model_params = {
    'ansEmbedSize': 300,
    'ansObjective': 'both',
    'commitment_cost': 0.25,
    'conditionEncoder': 'images',
    'decMode': 'RNN',
    'decay': 0.99,
    'decoderVar': 'gumbel',
    'discriminatorMode': False,
    'dropout': 0,
    'embedSize': 512,
    'encMode': 'RNN',
    'endToken': 5993,
    'imgEmbedSize': 512,
    'imgFeatureSize': 2048,
    'latentRecurrent': True,
    'latentSize': 64,
    'numLayers': 1,
    'num_ans_candidates': 3129,
    'num_embeddings': 4,
    'num_vars': 128,
    'policyMode': 'entangle',
    'poolAtten': 'hierarchical',
    'rnnHiddenSize': 512,
    'softStartIters': 10000,
    'startToken': 5992,
    'temp': 1.0,
    'tempAnnealRate': 3e-05,
    'tempMin': 0.5,
    'unkToken': 5991,
    'useImage': True,
    'vaeMode': 'gumbelst-vae',
    'vaeObjective': 'cond_vae_precc',
    'vocabSize': 5994,
    'wordDropoutRate': 0.3,
}

class FakeLogSoftmax(nn.Module):
    def __init__(self, dim, mode='min'):
        super(FakeLogSoftmax, self).__init__()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.dim = dim
        self.mode = mode

    def forward(self, x):
        r = self.logSoftmax(x)
        if self.mode == 'min':
            r[:, self.dim] = r.min() - 1
        elif self.mode == 'max':
            r[:, self.dim] = r.max() / 2
        return r

def test_decoder():
    qbot = utils.loadModel(model_params, agent='qbot')
    qbot.cuda()

    batch = {
        'img_pool': torch.randn([10, 2, 36, 2048]).cuda(),
        'ans_idx': torch.randint(3000, [10]).cuda(),
        'ans_rel': torch.randint(2, [10]).cuda(),
    }

    for arg in ['min', 'max']:
        #qbot.decoder.logSoftmax = FakeLogSoftmax(qbot.decoder.endToken, arg).cuda()
        for it in range(5):
            qbot.reset()
            qbot.observe(images=batch['img_pool'])
            qbot.observe(start_question=True)
            qbot.observe(start_answer=True)

            for i in range(10):
                ques, ques_len, stop_logits, dis_logit, ques_soft = qbot.forwardDecode(
                                                dec_inference='sample',
                                                z_inference='sample',
                                                z_source='policy',
                                                ret_soft=True)
                qbot.observe(ques=ques, ques_len=ques_len, ques_soft=ques_soft, gt_ques=False)
                qbot.observe(ans=batch['ans_idx'], ans_rel=batch['ans_rel'])


if __name__ == '__main__':
    test_decoder()

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQGModel(nn.Module):

    def __init__(self, params, decoder):
        super().__init__()

        self.params = params.copy()
        self.decoder = decoder

        embed_size = self.params['ansEmbedSize']
        self.ans_embed = nn.Linear(self.params['num_ans_candidates'], embed_size)
        self.ans_rel_embed = nn.Linear(2, embed_size)
        self.img_embed = nn.Linear(self.params['imgFeatureSize'], embed_size)

        self.dec_embed = nn.Linear(embed_size, self.params['embedSize'])


    def forward(self, ans, ans_rel, v_emb, gt_ques):
        # embed each then combine w/ sum
        ans = self.ans_embed(ans)
        ans_rel = self.ans_rel_embed(ans_rel)
        img = self.img_embed(v_emb)
        emb = ans + ans_rel + img
        # TODO: dropout?
        emb_dec = self.dec_embed(emb)

        log_probs = self.decoder(emb_dec, None, gt_ques)
        return log_probs


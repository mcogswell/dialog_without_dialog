import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.utils.weight_norm import weight_norm
from models.answerer import FCNet


class ImageSetContextCoder(nn.Module):
    def __init__(self, imgEmbedSize, imgFeatureSize, rnnHiddenSize):
        super(ImageSetContextCoder, self).__init__()
        # this should match the params['rnnHiddenSize'] passed to an RNNEncoder
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.rnnHiddenSize = rnnHiddenSize
        self.tracking = False

        self.imgAttnEmbed = FCNet([self.imgFeatureSize, 1024])
        self.queryEmbed = FCNet([self.rnnHiddenSize, 1024])

        self.linear = weight_norm(nn.Linear(1024, 1), dim=None)

        # model
        self.v_net = FCNet([self.imgFeatureSize, self.imgEmbedSize])

        # attention
        self.aveimgAttnEmbed = FCNet([self.imgFeatureSize, 1024])
        self.avequeryEmbed = FCNet([self.rnnHiddenSize, 1024])
        self.avelinear = weight_norm(nn.Linear(1024, 1), dim=None)

    def forward(self, imgs, query=None):
        # imgs: [n_batch x n_pool x n_boxes x feat_dim]
        batchSize, m, n, _ = imgs.shape

        # do the mean pooling over the image.
        v_proj = self.imgAttnEmbed(imgs)

        if query is not None:
            q_proj = self.queryEmbed(query)
            joint_repr = v_proj * q_proj.unsqueeze(1).unsqueeze(2)
        else:
            joint_repr = v_proj

        logits = self.linear(joint_repr)

        region_attn = F.softmax(logits, dim=2)

        v_mean_proj = self.aveimgAttnEmbed(imgs.mean(2))
        if query is not None:
            q_mean_proj = self.queryEmbed(query)
            joint_mean_repr = v_mean_proj * q_mean_proj.unsqueeze(1)
        else:
            joint_mean_repr = v_mean_proj

        logits = self.linear(joint_mean_repr)
        img_attn = F.softmax(logits, dim=1)
        attn = (img_attn.unsqueeze(3) * region_attn).view(batchSize, -1, 1)
        if self.tracking:
            self.attn = attn
            self.img_attn = img_attn
            self.region_attn = region_attn

        iEmbed = (attn * imgs.view(batchSize, -1, self.imgFeatureSize)).sum(1)
        iEmbed = self.v_net(iEmbed)

        return iEmbed



class ImageAttention(nn.Module):
    def __init__(self, params):
        super(ImageAttention, self).__init__()

        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.ansEmbedSize = params['ansEmbedSize']

        self.v_proj = FCNet([self.imgFeatureSize, 1024])
        self.q_proj = FCNet([self.rnnHiddenSize*2+self.ansEmbedSize, 1024])
        self.linear = weight_norm(nn.Linear(1024, 1), dim=None)

        # self.linear = nn.Linear(imgEmbedSize, 1)

    def forward(self, image, query):
        batch, m, n, _ = image.size()

        v_proj = self.v_proj(image) # [batch, k, qdim]
        q_proj = self.q_proj(query).unsqueeze(1).unsqueeze(2).repeat(1,m, n, 1)
        joint_repr = v_proj * q_proj
        logits = self.linear(joint_repr)
        attn = F.softmax(logits, dim=2)

        return attn

class ImageContextCoder(nn.Module):
    def __init__(self, params):
        super(ImageContextCoder, self).__init__()
        # this should match the params['rnnHiddenSize'] passed to an RNNEncoder
        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.ansEmbedSize = params['ansEmbedSize']
        self.tracking = False

        self.attention = ImageAttention(params)
        self.v_net = FCNet([self.imgFeatureSize, 1024])

    def forward(self, imgs, query):
        # imgs: [n_batch x n_pool x n_boxes x feat_dim]
        batchSize = imgs.shape[0]

        attn = self.attention(imgs, query)
        iEmbed = (attn * imgs).sum(2)
        iEmbed = self.v_net(iEmbed)

        if self.tracking:
            self.attn = attn
        return iEmbed

class FactAttention(nn.Module):
    def __init__(self, params):
        super(FactAttention, self).__init__()

        self.rnnHiddenSize = params['rnnHiddenSize']
        self.ansEmbedSize = params['ansEmbedSize']

        self.v_proj = FCNet([self.ansEmbedSize+self.rnnHiddenSize, 1024])
        self.q_proj = FCNet([self.rnnHiddenSize, 1024])
        self.linear = weight_norm(nn.Linear(1024, 1), dim=None)

        # self.linear = nn.Linear(imgEmbedSize, 1)

    def forward(self, fact, query):
        batch, m, _ = fact.size()

        v_proj = self.v_proj(fact) # [batch, k, qdim]
        q_proj = self.q_proj(query).unsqueeze(1).repeat(1,m,1)
        joint_repr = v_proj * q_proj
        logits = self.linear(joint_repr)
        attn = F.softmax(logits, dim=1)

        return attn


class FactContextCoder(nn.Module):
    def __init__(self, params):
        super(FactContextCoder, self).__init__()
        # this should match the params['rnnHiddenSize'] passed to an RNNEncoder
        self.imgEmbedSize = params['imgEmbedSize']
        self.imgFeatureSize = params['imgFeatureSize']
        self.rnnHiddenSize = params['rnnHiddenSize']
        self.ansEmbedSize = params['ansEmbedSize']

        self.attention = FactAttention(params)
        # TODO: add image attention when it's finished for RNNEncoder

    def forward(self, fact, query):
        # imgs: [n_batch x n_pool x n_boxes x feat_dim]
        batchSize = fact.shape[0]

        attn = self.attention(fact, query)
        fEmbed = (attn * fact).sum(1)
        # iEmbed = self.v_net(iEmbed)

        return fEmbed

import json
import os.path as pth
import base64
from io import BytesIO

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import PIL
import PIL.ImageDraw

from jinja2 import Template

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import models.questioner
import models.answerer

import misc.utilities as utils
import misc.dataloader


# COCO meta-data
# TODO: not portable
coco_path = 'datasets/coco/'
vqa_s3_path = 'https://vqa_mscoco_images.s3.amazonaws.com'
with open(pth.join(coco_path, 'coco.json'), 'r') as f:
    cocoj = json.load(f)
cid_to_file = {d['id']: d['file_path'] for d in cocoj['images']}

# CUB meta-data
cub_meta_path = 'data/cub_meta.json'
with open(cub_meta_path, 'r') as f:
    cub_meta = json.load(f)
cub_img_id_to_idx = {im['img_id']: idx for idx, im in enumerate(cub_meta['images'])}

# AWA meta-data
awa_meta_path = 'data/awa_meta.json'
with open(awa_meta_path, 'r') as f:
    awa_meta = json.load(f)
awa_img_id_to_idx = {im['img_id']: idx for idx, im in enumerate(awa_meta['images'])}

def load_image_path(img_id, size=None, dataset_name='VQA', local=True):
    if dataset_name == 'VQA':
        coco_id = img_id
        fname = cid_to_file[coco_id]
        if local:
            file = pth.join(coco_path, 'images', fname)
        else:
            file = '/'.join([vqa_s3_path, fname.lstrip('/')])
    elif dataset_name == 'CUB':
        img_idx = cub_img_id_to_idx[img_id]
        file = cub_meta['images'][img_idx]['img_path']
    elif dataset_name == 'AWA':
        img_idx = awa_img_id_to_idx[img_id]
        file = awa_meta['images'][img_idx]['img_path']
    return file

def load_image(img_id, size=None, dataset_name='VQA'):
    file = load_image_path(img_id, size=size, dataset_name=dataset_name)
    img = PIL.Image.open(file).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img, file

def img_to_datauri(fname=None, pil_image=None, torch_image=None, size=None):
    if fname:
        assert fname[-3:] == 'jpg'
        with open(fname, 'rb') as f:
            data = base64.b64encode(f.read())
    elif pil_image:
        buffer = BytesIO()
        pil_image.save(buffer, format='jpeg')
        data = base64.b64encode(buffer.getvalue())
    elif torch_image is not None:
        pil_image = PIL.Image.fromarray(torch_image.numpy())
        if size is not None:
            pil_image = pil_image.resize(size)
        return img_to_datauri(pil_image=pil_image)
    return 'data:image/jpeg;base64,{}'.format(data.decode('utf-8'))


def get_questions(batch, qbot, z_source, inference, ind2word, nsamples=1,
                  manualz=None):
    imgs = batch['img_pool']
    #imgs, ques = batch['img_pool'], None
    output = []
    z_inference, dec_inference = inference
    if z_source is not None and z_source[:6] == 'manual':
        assert len(manualz) == nsamples
    qbot.tracking = True
    for ns in range(nsamples):
        qbot.reset()
        qbot.observe(images=batch['img_pool'])
        qbot.observe(start_question=True)
        qbot.observe(start_answer=True)
        if 'ques' in batch:
            qbot.observe(ques=batch['ques'],
                         ques_len=batch['ques_len'],
                         gt_ques=True)
        if z_source[:6] == 'manual':
            zs = manualz[ns]
        else:
            zs = z_source
        gen = qbot.forwardDecode(dec_inference=dec_inference,
                                 z_inference=z_inference,
                                 z_source=zs)

        qidx, qlen, _ = gen
        z = qbot.z
        gen_ques_str = utils.old_idx_to_str(ind2word, qidx, qlen,
                                            batch['img_id_pool'], 0, [])
        gen_ques_str = [q[0].strip('<START> ') for q in gen_ques_str]
        gen_ques_str = [q.strip(' <END>') for q in gen_ques_str]
        output.append((gen_ques_str, show_z(z, qbot)))
    return list(zip(*output))


def single_dim_manual_z(zs, qbot, dim):
    K = qbot.num_embeddings
    V = qbot.num_vars
    K = min(K, 10)
    newz = []
    assert len(zs) == 1, 'start with greedy version'
    for samplei in range(len(zs)):
        newz.append([])
        for exi in range(len(zs[samplei])):
            z = zs[samplei][exi][1]
            z = z.reshape([V, -1])
            _, ix = z.max(dim=1, keepdim=True)
            onehotz = torch.zeros_like(z).scatter_(1, ix, 1.0)
            onehotz = onehotz.reshape(-1)
            newz[samplei].append(onehotz)
        newz[samplei] = torch.stack(newz[samplei])
    # nsamples x nbatch x qbot._num_embeddings
    newz = torch.stack(newz)
    newz = newz.reshape(newz.shape[:2] + (V, K))
    newz = newz.repeat(K, 1, 1, 1)
    for k in range(K):
        newz[k, :, dim, :] = 0
        newz[k, :, dim, k] = 1.0
    newz = newz.reshape(newz.shape[:2] + (V * K,))
    return newz


def interp_manual_z(z, replications=1, max_len=200):
    assert len(z) == 1
    samplei = 0
    targeti = 0
    z_paths = []
    for exi in range(len(z[0])):
        zsource = z[samplei][exi][1]
        ztarget = z[samplei][targeti][1]

        change_idxs = (ztarget - zsource).nonzero().sort()[0]
        change_idxs.shape

        z_list = [zsource]
        z_len = change_idxs.shape[0] + 1

        if z_len > max_len:
            print('z_len {} exceeds max_len {}'.format(z_len, max_len))

        for i in change_idxs:
            newz = z_list[-1].clone()
            newz[i] = ztarget[i]
            for _ in range(replications):
                z_list.append(newz)

        z_list = z_list + z_list[-1:] * (max_len - len(z_list))
        z_exi_path = torch.stack(z_list[:max_len])
        z_paths.append(z_exi_path[:, None, :])

    z_paths = torch.cat(z_paths, dim=1)
    return z_paths


def show_z(z, qbot):
    number_of_vars_to_show = 10
    z, _ = z
    if qbot.varType == 'cont':
        return [('{:.3f}'.format(z[0][i, :].mean().item()), z[0][i]) for i in range(z[0].shape[0])]
        #return [('z', z[0][i]) for i in range(z[0].shape[0])]
    K = qbot.num_embeddings
    V = qbot.num_vars
    vs = []
    for v in range(V):
        _, v_idx = z[:, v*K:(v+1)*K].max(dim=1)
        vs.append(v_idx)
    vs = torch.stack(vs, dim=0).t().tolist()
    return [('_'.join(['{:0>2d}'.format(iv) for iv in i_vars][:number_of_vars_to_show]) + '...', z[i]) for i, i_vars in enumerate(vs)]


def pool_atten_uris(img_attn, side_length=30, cm=plt.get_cmap('viridis'), wrap_period=4):
    N, ps = img_attn.shape
    nrows = (ps-1) // wrap_period + 1
    ncols = ps if nrows == 1 else wrap_period

    imgs = []
    for exi in range(N):
        img = PIL.Image.new('RGB', (ncols * side_length, nrows * side_length), 'black')
        draw = PIL.ImageDraw.Draw(img)
        for i in range(ps):
            row = i // wrap_period
            col = i % wrap_period
            att_color = tuple([int(255 * t) for t in cm(float(img_attn[exi][i]))])
            top_left = (col * side_length, row * side_length)
            bottom_right = ((col+1) * side_length, (row+1) * side_length)
            draw.rectangle([top_left, bottom_right], fill=att_color)
        img = img_to_datauri(pil_image=img)
        imgs.append(img)
    return imgs


def bottomup_heatmap_image(img, att, spatial, cm=plt.get_cmap('viridis')):
    global hi1
    # color the background according to the colormap because everything's too small to see
    img[:] = 255.
    H, W, _ = img.shape
    n_obj, = att.shape
    att_img = torch.zeros(img.shape)
    left =   (spatial[:, 0] * W).to(torch.int).clamp(0, W-1)
    top =    (spatial[:, 1] * H).to(torch.int).clamp(0, H-1)
    right =  (spatial[:, 2] * W).to(torch.int).clamp(0, W-1)
    bottom = (spatial[:, 3] * H).to(torch.int).clamp(0, H-1)
    for k in range(n_obj):
        t, b, l, r = top[k], bottom[k], left[k], right[k]
        # this version is just black and white without using a color map
        att_img[t:b, l:r] += att[k] * img[t:b, l:r].to(att.dtype)
    att_img = att_img[:, :, 0].detach().numpy() / 255
    att_img = 255 * torch.tensor(cm(att_img))[:, :, :3]
    return att_img.to(img.dtype)


def region_att_images(img_pool_ids, region_attn, img_pool_spatial, dataset_name):
    aimgs = []
    N, pool_size = img_pool_ids.shape
    for exi in range(N):
        aimgs.append([])
        for pi in range(pool_size):
            img, _ = load_image(img_pool_ids[exi][pi].item(), size=(50, 50),
                                dataset_name=dataset_name)
            img = torch.tensor(np.array(img))
            att = region_attn[exi, pi]
            spatial = img_pool_spatial[exi, pi]
            aimg = bottomup_heatmap_image(img, att, spatial)
            aimg = img_to_datauri(torch_image=aimg)
            aimgs[-1].append(aimg)
    return aimgs

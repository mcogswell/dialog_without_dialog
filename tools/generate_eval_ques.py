import os
import gc
import random
import pprint
from six.moves import range
from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
import time

import h5py
import numpy as np
import torch
import random
import json
import pdb
import pickle


def generate_eval_anno(imageid2ques, image_list, pool_neighbor, sample):
    out = {}
    annotations = []
    count = 0
    for i, image_id in enumerate(image_list):
        for j in range(len(pool_neighbor[i])):
            neighbor_imgid = image_list[int(pool_neighbor[i,j])]
            # append the question in the corresponding image id.
            for ques in imageid2ques[neighbor_imgid]:
                annotations.append({'image_id':image_id, 'caption':ques, 'id':count})
                count += 1

    out['annotations'] = annotations
    out['type'] = []
    out['licenses'] = []
    out['images'] = sample['images']
    out['info'] = []

    return out


COCO_VAL_CAPTION_PATH = 'data/coco/captions_val2014.json'
VQA_QUES_VAL_PATH = 'data/VQA/v2_OpenEnded_mscoco_val2014_questions.json'
POOL_INFO_PATH = 'data/pool_info_v4.pkl'

sample = json.load(open(COCO_VAL_CAPTION_PATH, 'r'))

# generate random question for fix pool during evaluations. Since each image has a unique pool id stored in pool_info.pkl

vqa_ques_val = json.load(open(VQA_QUES_VAL_PATH, 'r'))['questions']

vqa_imageid2ques = {}
for ques in vqa_ques_val:
    if ques['image_id'] not in vqa_imageid2ques:
        vqa_imageid2ques[ques['image_id']] = []
    vqa_imageid2ques[ques['image_id']].append(ques['question'])

pool_info = pickle.load(open(POOL_INFO_PATH, 'rb'))
val_image_list = pool_info['val1']['image_ids']

for size in [2, 4, 9]:
    for difficulty in ['easy', 'rand', 'hard']:
        pool = pool_info['val1'][difficulty]
        pool = np.concatenate([np.arange(len(pool))[:, None], pool], axis=1)
        anno = generate_eval_anno(vqa_imageid2ques, val_image_list, pool[:,:size], sample)
        json.dump(anno, open(f'coco-caption/annotations/dwd_{difficulty}_{size}.json', 'w'))

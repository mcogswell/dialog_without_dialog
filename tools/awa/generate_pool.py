import os
from six.moves import range

import numpy as np
import pymp
import pickle

import h5py
import numpy as np
from six import iteritems
from six.moves import range
import json
import pdb



# input
awa_info = 'data/awa_meta.json'
splits = ['train', 'val', 'test']
max_pool_size = 50
# output
cache_file = 'data/awa_pool_info_v1.pkl'

with open(awa_info, 'r') as f:
    awa_meta = json.load(f)
images = awa_meta['images']
img_id_to_img_idx = {im['img_id']: img_idx for img_idx, im in enumerate(images)}

num_pools_per_split = {
    'train': 50000,
    'val': 10000,
    'test': 20000,
}
image_ids_by_split = {}
class_list_by_split = {}
image_ids_by_class_by_split = {}
for split in splits:
    image_ids = [im['img_id'] for im in images if im['split'] == split]
    image_ids_by_split[split] = image_ids

    class_list = [im['class'] for im in images if im['split'] == split]
    class_list_by_split[split] = sorted(list(set(class_list)))

    image_ids_by_class_by_split[split] = {}
    for cls in class_list:
        image_ids = [im['img_id'] for im in images if im['split'] == split and im['class'] == cls]
        image_ids_by_class_by_split[split][cls] = image_ids

result = {}
for split in splits:
    # get features from h5 and normalize them
    image_ids = image_ids_by_split[split]
    class_list = class_list_by_split[split]
    num_split = num_pools_per_split[split]

    # sample random pools
    np.random.seed(8)
    rand_pool = np.zeros((num_split, max_pool_size))
    for i in range(num_split):
        class_list = np.random.permutation(class_list)
        pool = []
        for cls in class_list[:max_pool_size]:
            cls_img_ids = image_ids_by_class_by_split[split][cls]
            img_id = np.random.choice(cls_img_ids)
            pool.append(img_id)
        # NOTE: In VQA each pool has a root image because it's attached to one question,
        # so only n-1 of the candidates from the pre-generated pools are used.
        # Here there is no root image, so n of the candidates are used
        rand_pool[i] = pool

    result[split] = {
        'rand': rand_pool,
        'image_ids': image_ids,
    }

# save the pool info
with open(cache_file, 'wb') as f:
    pickle.dump(result, f)

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



def get_neighbors(image_list, image_feature):
    
    total_image = len(image_list)
    batch_size = 1
    p_length = int(total_image / batch_size)
    easy_pool = pymp.shared.array((total_image,100))
    hard_pool = pymp.shared.array((total_image,100))

    with pymp.Parallel(40) as p:
        for index in p.range(0, p_length):
            diff = image_feature - image_feature[index]
            L2_dis = np.sqrt(np.sum(pow(diff, 2), 1) / 2048)
            # sort the l2 distance.
            hard_pool[index] = np.argsort(L2_dis)[1:101] 
            easy_pool[index] = np.argsort(-L2_dis)[:100]
            print('finish worker', index)

    rand_pool = np.zeros((total_image, 100))
    for i in range(total_image):
        rand_idx = np.random.permutation(total_image)
        rand_pool[i] = rand_idx[:100] #[val_image_list[rand_idx[i]] for i in range(100)]
    
    return hard_pool, easy_pool, rand_pool

# input
inputImg = 'data/img_bottom_up.h5'
inputQues = 'data/v2_vqa_data.h5'
inputJson = 'data/v2_vqa_info.json'
splitInfo = 'data/split_info_v3.json'
# output
cache_file = 'data/pool_info_v3.pkl'

data = {}
result = {}

print('\nDataloader loading json file: ' + inputJson)
with open(inputJson, 'r') as fileId:
    info = json.load(fileId)
    # Absorb values
    for key, value in iteritems(info):
        data[key] = value

# load split info (using custom train/val1/val2 splits)
print('\nLoading splits from: ' + splitInfo)
with open(splitInfo, 'r') as f:
    split_info = json.load(f)
img_id2split = {}
image_ids_by_split = split_info['split_img_ids']
for split in split_info['splits']:
    num_split = len(image_ids_by_split[split])
    print(f'{split}: {num_split}')
    for img_id in split_info['split_img_ids'][split]:
        img_id2split[img_id] = split

# load the image feature
print('Loading image features')
imgFile = h5py.File(inputImg, 'r')
im_fv = imgFile['images_train'][:]

# compute pools
image_feature_by_split = {}
for split in split_info['splits']:
    # get features from h5 and normalize them
    image_ids = image_ids_by_split[split]
    num_split = len(image_ids)
    image_feature = np.zeros([num_split, 2048])
    image_feature_by_split[split] = image_feature
    for i, image_id in enumerate(image_ids):
        feature = im_fv[data['unique_image'][str(image_id)]]
        image_feature[i] = feature.sum(0) / 36

    # find closest / farthest / distance unaware neighbors
    np.random.seed(8)
    print(f'computing neighbors for {split}')
    hard_pool, easy_pool, rand_pool = get_neighbors(image_ids, image_feature)
    result[split] = {
        'hard': hard_pool,
        'easy': easy_pool,
        'rand': rand_pool,
        'image_ids': image_ids,
    }

# save the pool info
with open(cache_file, 'wb') as f:
    pickle.dump(result, f)

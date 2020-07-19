# The random pools in pool_info were created by uniformly randomly sampling from
# all images in a particular split. Sometimes this means the list of 100 images
# in a random pool contains the root image itself. In order to avoid that, this
# script simply moves the duplicate to the back of the list.
import pickle as pkl
import numpy as np


POOL_INFO_PATH = 'data/pool_info_v3.pkl'
NEW_POOL_INFO_PATH = 'data/pool_info_v4.pkl'

with open(POOL_INFO_PATH, 'rb') as f:
    pool_info = pkl.load(f)

for split in pool_info:
    img_ids = pool_info[split]['image_ids']
    for difficulty in ['easy', 'rand', 'hard']:
        pool = pool_info[split][difficulty]
        for img_idx, img_id in enumerate(img_ids):
            pool_idxs = pool[img_idx]
            # check if there's a duplicate
            if (pool_idxs == img_idx).sum() != 0:
                # if there is then move it to the back
                dup_idxs = np.where(pool_idxs == img_idx)[0]
                assert len(dup_idxs) == 1, 'easy to deal with, but unlikely'
                dup_idx = dup_idxs[0]
                new_idxs = np.concatenate([pool_idxs[:dup_idx],
                                           pool_idxs[dup_idx+1:],
                                           [img_idx]])
                pool[img_idx] = new_idxs


with open(NEW_POOL_INFO_PATH, 'wb') as f:
    pkl.dump(pool_info, f)

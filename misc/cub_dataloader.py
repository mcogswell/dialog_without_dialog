import json
import pickle as pkl

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

class CUBPoolDataset(Dataset):
    def __init__(self, params, splits, load_vis_info=False):
        '''
        This dataset is similar to the VQAPoolDataset. It loads pools of
        CUB images in different configurations.

        poolSize - Number of images in the pool. Must be 2 for contrast
                   and contrast-random.

        On indexing (similar to VQA):
        Images and pools need to be indexed. Each of these has
        two indices, one corresponding to some external source which is
        probably not in range(0, num_items) and another which is for internal
        use and is in range(0, num_items). The latter use `_id` and the
        former `_idx`. This is even more complicated because some things
        (pools) have different indices for each split. Here's a summary:

            img_id: CUB image id
            img_idx: internal index in range(0, num_images)
            one per split:
                pool_id: pool id (indexes a whole pool, not images inside)
                pool_idx: pool index in range(0, num_pools=len(self))
                pool_info_idx: one per image; in range(0, num_images_in_split)

        '''
        self.params = params
        self.pool_info_path = params['poolInfo']
        self.input_img_path = params['inputImg']
        self.name = 'CUB'

        # CUB meta data
        print('Loading meta-data from', params['cubInfo'])
        with open(params['cubInfo'], 'r') as f:
            self.cub_meta = json.load(f)
        self.all_images = self.cub_meta['images']
        self.img_id_to_idx = {im['img_id']: idx for idx, im in enumerate(self.all_images)}
        self.splits = splits
        self._split = splits[0]

        # pool configuration
        self.pool_size = params['poolSize']
        self.pool_type = 'rand'

        print('Loading the pool from', self.pool_info_path)
        with open(self.pool_info_path, 'rb') as f:
            self.pool_info = pkl.load(f)

        self.pool_idxs_by_split = {}
        for split in self.splits:
            pool_idxs = list(range(len(self.pool_info[split]['rand'])))
            self.pool_idxs_by_split[split] = pool_idxs

        print('Dataloader loading h5 file: ' + self.input_img_path)
        self._loaded_h5 = False
        #self._load_h5_data()

        if load_vis_info:
            self._load_vis_info(params)

    def _load_vis_info(self, params):
        # quick hack to inclue necessary information from VQA for visualization
        inputJson = params['inputJson']
        print('\nDataloader (vis_info) loading vqa json file: ' + inputJson)
        with open(inputJson, 'r') as fileId:
            info = json.load(fileId)
        # question vocab
        self.vocab = info['vocab']
        wordCount = len(self.vocab)
        self.word2ind = {w:i+1 for i, w in enumerate(self.vocab)}
        self.startToken = self.word2ind['<START>'] 
        self.endToken = self.word2ind['<END>'] 
        self.unkToken = self.word2ind['<UNK>']
        self.vocabSize = wordCount + 1 # Padding token is at index 0
        print(f'Vocab size with <START>, <END>: {self.vocabSize}')
        self.ind2word = {
            int(ind): word
            for word, ind in self.word2ind.items()
        }
        self.ans2label = info['ans2label']

    def _load_h5_data(self):
        #print('loading....')
        self._loaded_h5 = True

        imgFile = h5py.File(self.input_img_path, 'r')
        # indexed by img_idx
        if self.params.get('useRedis', 0):
            from misc.memory_store.redis_dataset import RedisDataset
            self.im_fv = RedisDataset(key=self.input_img_path + '_image_features',
                                      dataset=imgFile['image_features'])
            self.im_spatial = RedisDataset(key=self.input_img_path + '_spatial_features',
                                           dataset=imgFile['spatial_features'])
        else:
            self.im_fv = imgFile['image_features']
            self.im_spatial = imgFile['spatial_features']

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.splits
        self._split = split

    def __len__(self):
        return len(self.pool_info[self.split][self.pool_type])

    def __getitem__(self, pool_idx):
        # NOTE: This needs to happen here and NOT in __init__() because
        # different h5py.File objects should be created in each worker
        # spawned by the Dataloader. (Otherwise behavior is undefined.)
        # This also allows for faster data loading.
        if not self._loaded_h5:
            self._load_h5_data()
        if self.split != 'train':
            np.random.seed(pool_idx)

        # figure out which images to put in the pool and put them there
        candidates = self.pool_info[self.split][self.pool_type][pool_idx]
        pool_img_ids = candidates[:self.pool_size]
        img_id_pool = np.zeros(self.pool_size)
        img_pool = np.zeros([self.pool_size, 36, 2048])
        img_pool_spatial = np.zeros([self.pool_size, 36, 6])
        target_id = int(pool_img_ids[0])
        np.random.shuffle(pool_img_ids)
        for i, pool_img_id in enumerate(pool_img_ids):
            pool_img_idx = self.img_id_to_idx[pool_img_id]
            img_id_pool[i] = pool_img_id
            img_pool[i] = self.im_fv[pool_img_idx]
            img_pool_spatial[i] = self.im_spatial[pool_img_idx]
            if pool_img_id == target_id:
                target_pool = i
        target_idx = self.img_id_to_idx[target_id]
        target_image = self.im_fv[target_idx]

        item = {}
        item['img_pool'] = torch.from_numpy(img_pool).float()
        item['img_pool_spatial'] = torch.from_numpy(img_pool_spatial).float()
        item['target_pool'] = torch.LongTensor(1).fill_(target_pool)
        item['img_id_pool'] =  torch.from_numpy(img_id_pool).long()
        item['target_image'] =  torch.from_numpy(target_image)
        return item

import os
import json
import h5py
import numpy as np
import torch
import random
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import _pickle as cPickle

class VQAPoolDataset(Dataset):
    def __init__(self, params, splits):
        '''
        This dataset loads pools of images in different configurations.
        Sometimes the pools have questions from VQA paired with them.

        poolType - There are 5 options

        poolSize - Number of images in the pool. Must be 2 for contrast
                   and contrast-random.

        On indexing:
        Images, questions, and pools all need to be indexed. Each of these has
        two indices, one corresponding to some external source which is
        probably not in range(0, num_items) and another which is for internal
        use and is in range(0, num_items). The latter use `_id` and the
        former `_idx`. This is even more complicated because some things
        (pools) have different indices for each split. Here's a summary:

            ques_id: vqa question id
            ques_idx: internal question index in range(0, num_questions)
            img_id: coco/vqa image id
            img_idx: internal index in range(0, num_images)
            one per split:
                pool_id: pool id (indexes a whole pool, not images inside)
                pool_idx: pool index in range(0, num_pools=len(self))
                pool_info_idx: one per image; in range(0, num_images_in_split)

        '''
        self.params = params
        self.name = 'VQA'

        # data file locations
        self.inputImg = params['inputImg']
        self.inputQues = params['inputQues']
        self.inputJson = params['inputJson']
        self.splitInfo = params.get('splitInfo', None)
        self.poolInfo = params.get('poolInfo', None)
        # pool configuration
        self.poolType = params['poolType']
        self.poolSize = params['poolSize']
        if self.poolType == 'contrast':
            # contrast is equivalent to contrast-random at poolSize 2 or below
            assert self.poolSize <= 2, 'Use contrast-random for poolSize > 2'
        # whether to include a random question (for abot relevance training)
        self.randQues = params['randQues']

        print('\nDataloader loading json file: ' + self.inputJson)
        with open(self.inputJson, 'r') as fileId:
            info = json.load(fileId)

        # question list
        # TODO: Yes, the key in the json is wrong. It's really a list of
        # questions... refactor the pre-processing before release?
        self.questions = info['imgs']
        self.ques_id2idx = {
            ques['question_id']: ques_idx
            for ques_idx, ques in enumerate(self.questions)
        }
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
        # image list
        self.img_id2idx = {
            int(img_id_str): img_idx
            for img_id_str, img_idx in info['unique_image'].items()
        }
        self.img_idx2id = { v: k for k, v in self.img_id2idx.items() }
        # answer vocab
        self.ans2label = info['ans2label']
        self.num_ans_candidates = len(self.ans2label)
        self.ans_uncertain_token = self.num_ans_candidates
        print('number of answer candidates: ' + str(self.num_ans_candidates))

        # load split information
        print('Split info file: ' + self.splitInfo)
        if self.splitInfo is not None:
            with open(self.splitInfo, 'r') as f:
                split_info = json.load(f)
            self.split_info = split_info
            split_names = set(split_info['splits'])
            self.img_id2split = {}
            for split in split_names:
                for img_id in split_info['split_img_ids'][split]:
                    self.img_id2split[img_id] = split
                # make sure no images are shared between splits
                for split2 in split_names:
                    if split2 != split:
                        img_ids1 = split_info['split_img_ids'][split]
                        img_ids2 = split_info['split_img_ids'][split2]
                        assert len(set(img_ids1).intersection(img_ids2)) == 0
        else:
            self.split_info = None
            split_names = set(['train', 'val'])
        assert set(splits) <= split_names
        self.splits = tuple(splits)
        self.split = 'train'

        # initialize pool stuff
        self.pool_ids_by_split = {}
        self.img_id2pool_id_by_split = {}
        for split in self.splits: 
            self.pool_ids_by_split[split] = []
            self.img_id2pool_id_by_split[split] = {}

        # assign a pool_id to each pool in each split
        # NOTE: Each question forms the base of exactly one pool, though
        # some questions can be excluded.
        for ques_idx in range(len(self.questions)):
            ques = self.questions[ques_idx]
            split = self._get_split(ques)
            pool_ids = self.pool_ids_by_split[split]
            if self.poolType == 'contrast' and self.poolSize > 1:
                diff_pool = ques['ques_with_diff_ans']
                if len(diff_pool) >= 1:
                    # sometimes ques_idx becomes pool_id, but only for some ques
                    pool_ids.append(ques_idx)
            elif self.poolType == 'contrast-random':
                pool_ids.append(ques_idx)
            elif self.poolType in ['random', 'hard', 'easy']:
                img_id2pool_id = self.img_id2pool_id_by_split[split]
                img_id = ques['image_id']
                # only one pool per root image
                if img_id not in img_id2pool_id:
                    pool_id = ques_idx
                    pool_ids.append(pool_id)
                    img_id2pool_id[img_id] = pool_id
        # assumes pool_id == ques_idx, which holds because of previous loop
        self.pool_idx2ques_idx_by_split = {}
        for split in self.splits: 
            pool_ids = self.pool_ids_by_split[split]
            self.pool_idx2ques_idx_by_split[split] = {
                pool_idx: pool_id # == ques_idx
                for pool_idx, pool_id in enumerate(pool_ids)
            }

        print('\nLoading the pool')
        with open(self.poolInfo, 'rb') as f:
            self.pool_info = cPickle.load(f)
        self.img_id2pool_info_idx = {}
        for split in self.splits:
            image_ids = self.pool_info[split]['image_ids']
            self.img_id2pool_info_idx[split] = {
                img_id: pool_info_idx
                for pool_info_idx, img_id in enumerate(image_ids)
            }

        print('\nDataloader loading Ques file: ' + self.inputQues)
        print('Dataloader loading h5 file: ' + self.inputImg)
        self._loaded_h5 = False
        #self._load_h5_data()

    def _get_split(self, ques):
        if self.split_info is None:
            return ques['split']
        else:
            return self.img_id2split[ques['image_id']]

    def _load_h5_data(self):
        #print('loading....')
        self._loaded_h5 = True
        quesFile = h5py.File(self.inputQues, 'r')
        # indexed by ques_idx
        self.ques = np.array(quesFile['ques'][:], dtype='int32')
        self.quesLen = np.array(quesFile['ques_len'][:], dtype='int32')
        quesFile.close()

        file_params = {
            # rough dataset size
            'rdcc_nbytes': None, #1 * 1024**3,
            # 10 * num_examples
            'rdcc_nslots': None, #1232870,
            'swmr': False,
        }
        imgFile = h5py.File(self.inputImg, 'r', **file_params)
        # indexed by img_idx
        if self.params.get('useRedis', 0):
            from misc.memory_store.redis_dataset import RedisDataset
            self.im_fv = RedisDataset(key=self.inputImg + '_images_train',
                                      dataset=imgFile['images_train'])
            self.im_spatial = RedisDataset(key=self.inputImg + '_spatial_features_train',
                                           dataset=imgFile['spatial_features_train'])
        else:
            self.im_fv = imgFile['images_train']
            self.im_spatial = imgFile['spatial_features_train']

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.splits
        self._split = split

    def __len__(self):
        return len(self.pool_ids_by_split[self.split])

    def __getitem__(self, pool_idx):
        # NOTE: This needs to happen here and NOT in __init__() because
        # different h5py.File objects should be created in each worker
        # spawned by the Dataloader. (Otherwise behavior is undefined.)
        # This also allows for faster data loading.
        if not self._loaded_h5:
            self._load_h5_data()
        if self.split != 'train':
            np.random.seed(pool_idx)

        # There are a lot of different indices here. See the docstring for
        # __init__ to learn what they all mean.
        pool_ids = self.pool_ids_by_split[self.split]
        pool_id = pool_ids[pool_idx]
        pool_idx2ques_idx = self.pool_idx2ques_idx_by_split[self.split]
        ques_idx = pool_idx2ques_idx[pool_idx]
        ques = self.questions[ques_idx]
        ques_id = ques['question_id']
        img_id = ques['image_id'] 
        img_idx = self.img_id2idx[img_id]
        item = {'pool_id': pool_id}

        # figure out which images to put in the pool and put them there
        pool_img_idxs = self._pool_img_idxs(ques)
        img_id_pool = np.zeros(self.poolSize)
        img_pool = np.zeros([self.poolSize, 36, 2048])
        img_pool_spatial = np.zeros([self.poolSize, 36, 6])
        np.random.shuffle(pool_img_idxs)
        for i, pool_img_idx in enumerate(pool_img_idxs):
            img_id_pool[i] = self.img_idx2id[pool_img_idx]
            img_pool[i] = self.im_fv[pool_img_idx]
            img_pool_spatial[i] = self.im_spatial[pool_img_idx]
            if pool_img_idx == img_idx:
                target_pool = i

        # ground truth answers
        labels = torch.from_numpy(np.array(ques['labels'])).long()
        scores = torch.from_numpy(np.array(ques['scores'], dtype=np.float32))
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        if len(labels) == 0:
            ansIdx = self.ans_uncertain_token
        else:
            ansIdx = labels[np.random.randint(0, len(labels))]

        # include a random question for training abot relevance
        if self.randQues:
            pool_ids = self.pool_ids_by_split[self.split]
            while True:
                rand_pool_idx = np.random.randint(len(pool_ids))
                rand_ques_idx = pool_idx2ques_idx[rand_pool_idx]
                rand_img_id = self.questions[rand_ques_idx]['image_id']
                rand_img_idx = self.img_id2idx[rand_img_id]
                if rand_img_idx not in pool_img_idxs:
                    break
            item['rand_ques'] = torch.from_numpy(self.ques[rand_ques_idx]).long()
            item['rand_ques_len'] = torch.LongTensor(1).fill_(self.quesLen[rand_ques_idx])

        target_image = self.im_fv[img_idx]

        item['ques'] = torch.from_numpy(self.ques[ques_idx]).long()
        item['ques_len'] = torch.LongTensor(1).fill_(self.quesLen[ques_idx])
        item['ans'] = target
        item['ansIdx'] = torch.LongTensor(1).fill_(ansIdx)
        item['img_pool'] = torch.from_numpy(img_pool).float()
        item['img_pool_spatial'] = torch.from_numpy(img_pool_spatial).float()
        item['target_pool'] = torch.LongTensor(1).fill_(target_pool)
        item['img_id_pool'] =  torch.from_numpy(img_id_pool).long()
        item['img_id'] = torch.LongTensor(1).fill_(img_id)
        item['target_image'] =  torch.from_numpy(target_image)
        item['do_stop_target'] =  torch.LongTensor(1).fill_(1)
        item['no_stop_target'] =  torch.LongTensor(1).fill_(0)
        return item

    def _pool_img_idxs(self, base_ques):
        '''Figure out which images to put in the pool.'''
        img_id = base_ques['image_id']
        img_idx = self.img_id2idx[img_id]
        pool_img_idxs = [img_idx]
        pool_idx2ques_idx = self.pool_idx2ques_idx_by_split[self.split]
        # only used for contrast pools
        diff_ques_ids = base_ques['ques_with_diff_ans']
        for i in range(self.poolSize-1):
            if self.poolType in ['contrast', 'contrast-random']:
                if i < len(diff_ques_ids):
                    rand_tmp = np.random.randint(len(diff_ques_ids))
                    rand_ques_id = diff_ques_ids[rand_tmp]
                    rand_ques_idx = self.ques_id2idx[rand_ques_id]
                    rand_img_id = self.questions[rand_ques_idx]['image_id']
                    rand_img_idx = self.img_id2idx[rand_img_id]
                    pool_img_idxs.append(rand_img_idx)
                else:
                    if self.poolType == 'contrast':
                        pool_img_idxs.append(img_idx) # duplicate the image
                    elif self.poolType == 'contrast-random':
                        pool_ids = self.pool_ids_by_split[self.split]
                        rand_pool_idx = np.random.randint(len(pool_ids))
                        rand_ques_idx = pool_idx2ques_idx[rand_pool_idx]
                        rand_img_id = self.questions[rand_ques_idx]['image_id']
                        rand_img_idx = self.img_id2idx[rand_img_id]
                        pool_img_idxs.append(rand_img_idx)

            elif self.split == 'train' and self.poolType == 'random':
                pool_ids = self.pool_ids_by_split[self.split]
                rand_pool_idx = np.random.randint(len(pool_ids))
                rand_ques_idx = pool_idx2ques_idx[rand_pool_idx]
                rand_img_id = self.questions[rand_ques_idx]['image_id']
                rand_img_idx = self.img_id2idx[rand_img_id]
                pool_img_idxs.append(rand_img_idx)

            elif self.poolType in ['easy', 'random', 'hard']:
                poolType = self.poolType if self.poolType != 'random' else 'rand'
                pool = self.pool_info[self.split][poolType]
                pool_info_idx = self.img_id2pool_info_idx[self.split][img_id]
                candidates = pool[pool_info_idx]
                pool_img_idxs.append(candidates[i])

        return pool_img_idxs

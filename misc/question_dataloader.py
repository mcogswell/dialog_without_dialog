import os
import json
import h5py
import numpy as np
import torch
import random
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import _pickle as cPickle

class VQAQuestionDataset(Dataset):
    def __init__(self, params, splits):
        '''
        This dataset loads VQA questions.

            ques_id: vqa question id
            ques_idx: internal index for questions from all splits range(0, N)
            ques_split_idx: range(0, num questions in this split == dataset len)
        '''
        self.params = params
        self.name = 'VQAQues'

        # data file locations
        self.inputQues = params['inputQues']
        self.inputJson = params['inputJson']
        self.splitInfo = params.get('splitInfo', None)

        print('\nDataloader loading json file: ' + self.inputJson)
        with open(self.inputJson, 'r') as fileId:
            info = json.load(fileId)

        # question list
        # Yes, the key in the json is wrong. It's really a list of questions
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
        # image list (this is needed because splits are based on img_id)
        self.img_id2idx = {
            int(img_id_str): img_idx
            for img_id_str, img_idx in info['unique_image'].items()
        }
        self.img_idx2id = { v: k for k, v in self.img_id2idx.items() }

        # load split information
        print('Split info file: ' + self.splitInfo)
        with open(self.splitInfo, 'r') as f:
            split_info = json.load(f)
        self.split_info = split_info
        split_names = set(split_info['splits'])
        self.img_id2split = {}
        for split in split_names:
            for img_id in split_info['split_img_ids'][split]:
                self.img_id2split[img_id] = split
        assert set(splits) <= split_names
        self.splits = tuple(splits)
        self.split = 'train'
        self.questions_by_split = {split: [] for split in self.splits}
        self.q_split_idx2idx = {}
        for ques_idx, ques in enumerate(self.questions):
            ques_split = self._get_split(ques)
            if ques_split in self.splits:
                ques_split_idx = len(self.questions_by_split[ques_split])
                self.q_split_idx2idx[ques_split_idx] = ques_idx
                self.questions_by_split[ques_split].append(ques)

        print('\nDataloader loading Ques file: ' + self.inputQues)
        quesFile = h5py.File(self.inputQues, 'r')
        # indexed by ques_idx
        self.ques = np.array(quesFile['ques'][:], dtype='int32')
        self.quesLen = np.array(quesFile['ques_len'][:], dtype='int32')
        quesFile.close()

    def _get_split(self, ques):
        return self.img_id2split[ques['image_id']]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.splits
        self._split = split

    def __len__(self):
        return len(self.questions_by_split[self.split])

    def __getitem__(self, ques_split_idx):
        if self.split != 'train':
            np.random.seed(ques_split_idx)

        ques = self.questions_by_split[self.split][ques_split_idx]
        ques_id = ques['question_id']
        item = {}

        ques_idx = self.q_split_idx2idx[ques_split_idx]
        item['ques'] = torch.from_numpy(self.ques[ques_idx]).long()
        item['ques_len'] = torch.LongTensor(1).fill_(self.quesLen[ques_idx])
        return item

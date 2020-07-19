import os
import json
import h5py
import copy
import argparse
import numpy as np
import pickle
import pdb


inputJson = 'v2_vqa_info.json'
bottomupJson = 'train36_imgid2idx.pkl'
imagePath = 'train36.hdf5'

info = json.load(open(inputJson, 'r'))
bottomupJson = pickle.load(open(bottomupJson,'rb'))
imgFile = h5py.File(imagePath, 'r')
image_features = imgFile['image_features'][:]
image_bb = imgFile['image_bb'][:]
spatial_features = imgFile['spatial_features'][:]

h_train = h5py.File('img_bottom_up.h5', "w")


train_img_features = h_train.create_dataset(
        'images_train', (len(info['unique_image']), 36, 2048), 'f')
train_img_bb = h_train.create_dataset(
        'images_bb_train', (len(info['unique_image']), 36, 4), 'f')
train_spatial_img_features = h_train.create_dataset(
        'spatial_features_train', (len(info['unique_image']), 36, 6), 'f')

train_counter = 0

for img in info['unique_image']:
    image_id = int(img)
    idx = bottomupJson[image_id]
    train_img_features[train_counter,:,:] = image_features[idx]
    train_img_bb[train_counter,:,:] = image_bb[idx]
    train_spatial_img_features[train_counter,:,:] = spatial_features[idx]
    train_counter += 1


print(train_counter)
h_train.close()
print("done!")

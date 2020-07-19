import argparse
import os.path as pth
import json
import random


parser = argparse.ArgumentParser()
parser.add_argument('--cub-dir', default='datasets/cub200/CUB_200_2011/')
parser.add_argument('--out-file', default='data/cub_meta.json')
args = parser.parse_args()

cub_dir = args.cub_dir


f_images = open(pth.join(cub_dir, 'images.txt'), 'r')
f_train_test_split = open(pth.join(cub_dir, 'train_test_split.txt'), 'r')
i_train_test_split = iter(f_train_test_split)
f_image_class_labels = open(pth.join(cub_dir, 'image_class_labels.txt'), 'r')
i_image_class_labels = iter(f_image_class_labels)


images = []

random.seed(8)

for l_images in f_images:
    l_split = next(f_train_test_split)
    l_label = next(f_image_class_labels)

    img_id, img_name = l_images.strip().split()
    _id, split_id = l_split.strip().split()
    assert _id == img_id
    _id, label = l_label.strip().split()
    assert _id == img_id
    split_id = int(split_id)
    img_id = int(img_id)
    label = int(label)

    if split_id == 0:
        if random.random() < 0.2:
            split = 'val'
        else:
            split = 'train' 
    else:
        split = 'test'

    images.append({
        'img_id': img_id,
        'img_name': img_name,
        'img_path': pth.join(cub_dir, 'images', img_name), # fully qualified
        'split': split,
        'label': label,
    })

print('train ', sum([im['split'] == 'train' for im in images]))
print('val ', sum([im['split'] == 'val' for im in images]))
print('test ', sum([im['split'] == 'test' for im in images]))


f_images.close()
f_train_test_split.close()
f_image_class_labels.close()

with open(pth.join(cub_dir, 'classes.txt'), 'r') as f:
    label_to_class_name = {}
    for l in f:
        label, class_name = l.strip().split()
        label_to_class_name[label] = class_name

with open(args.out_file, 'w') as f:
    json.dump({
        'images': images,
        'label_to_class_name': label_to_class_name,
    }, f)


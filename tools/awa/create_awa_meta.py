import argparse
import os.path as pth
import json
import glob
import random


parser = argparse.ArgumentParser()
parser.add_argument('--awa-dir', default='datasets/Animals_with_Attributes2/')
parser.add_argument('--out-file', default='data/awa_meta.json')
args = parser.parse_args()

awa_dir = args.awa_dir


with open(pth.join(awa_dir, 'classes.txt'), 'r') as f:
    classes = [l.strip().split()[1] for l in f]
class_name_to_id = {cls: idx for idx, cls in enumerate(classes)}
class_id_to_name = {class_name_to_id[cls]: cls for cls in class_name_to_id}

images = []
random.seed(8)
def sample_split():
    x = random.random()
    if x < 0.6:
        return 'train'
    elif x < 0.75:
        return 'val'
    else:
        return 'test'

for cls in classes:
    img_path_lst = glob.glob(pth.join(awa_dir, 'JPEGImages', cls, '*'))
    img_path_lst = sorted(list(set(img_path_lst)))
    for img_path in img_path_lst:
        fname = pth.basename(img_path)
        images.append({
            'class': cls,
            'img_path': img_path,
            'img_fname': fname,
            'img_name': fname.strip('.jpg'),
            'img_id': len(images),
            'split': sample_split(),
        })

assert len(set([im['img_id'] for im in images])) == len(images)

print('train ', sum([im['split'] == 'train' for im in images]))
print('val ', sum([im['split'] == 'val' for im in images]))
print('test ', sum([im['split'] == 'test' for im in images]))

with open(args.out_file, 'w') as f:
    json.dump({
        'images': images,
        'class_id_to_name': class_id_to_name,
    }, f)


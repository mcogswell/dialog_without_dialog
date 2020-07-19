import os.path as pth
import json
import numpy as np


data_dir = 'data/'
inputJson = pth.join(data_dir, 'v2_vqa_info.json')

# load vqa info and find indices of train/val images
print(f'Loading vqa v2 info from: {inputJson}')
with open(inputJson, 'r') as f:
    v2_info = json.load(f)
questions = v2_info['imgs'] # yes, this is a list of questions
splits_by_ques = [ques['split'] for ques in questions]
val_ixs = [ix for ix, split in enumerate(splits_by_ques) if split == 'val']
train_ixs = [ix for ix, split in enumerate(splits_by_ques) if split == 'train']
assert len(val_ixs + train_ixs) == len(questions)

# Split the val set into val1 (validation) and val2 (test) so paired images
# are kept together.
np.random.seed(8)
# validation, val1 - 30%
# test, val2 - 70%
ques_id2ix = {questions[ix]['question_id']: ix for ix in val_ixs}
val_img_ids = set(questions[ix]['image_id'] for ix in val_ixs)
val1_percent = 0.3
val1_len = int(len(val_ixs) * val1_percent)
val1_ixs = set()
val2_ixs = set()
val1_img_ids = set()
val2_img_ids = set()
def is_assigned(ix):
    # Figure out if a question is assigned to a split.
    # This also tracks question ixs which have already been assigned by
    # another question with the same image id.
    img_id = questions[ix]['image_id']
    assigned = False
    if img_id in val1_img_ids:
        assigned = True
        val1_ixs.add(ix)
    elif img_id in val2_img_ids:
        assigned = True
        val2_ixs.add(ix)
    return assigned

#def add_neighbors(ques, ixs, img_ids):
#    for ques_id in ques['ques_with_diff_ans']:
#        alt_ix = ques_id2ix[ques_id]
#        if is_assigned(alt_ix):
#            continue
#        alt_img_id = questions[alt_ix]['image_id']
#        ixs.add(alt_ix)
#        img_ids.add(alt_img_id)

# If val1 is filled up first then it might get a better selection of
# which questions are available, potentially biasing val2's choice.
for i, ix in enumerate(val_ixs):
    if is_assigned(ix):
        continue
    if np.random.rand() < val1_percent:
        ixs = val1_ixs
        img_ids = val1_img_ids
    else:
        ixs = val2_ixs
        img_ids = val2_img_ids
    ques = questions[ix]
    img_id = ques['image_id']
    ixs.add(ix)
    img_ids.add(img_id)
    #add_neighbors(ques, ixs, img_ids)

# how did the sampling procedure turn out?
ratio = len(val1_ixs) / (len(val1_ixs) + len(val2_ixs))
print(f'val1 percent goal: {val1_percent}')
print(f'questions: val1 / (val1 + val2) = {ratio}')
ratio = len(val1_img_ids) / (len(val1_img_ids) + len(val2_img_ids))
print(f'val1 percent goal: {val1_percent}')
print(f'images: val1 / (val1 + val2) = {ratio}')
assert len(val1_ixs) + len(val2_ixs) == len(val_ixs)
assert len(val1_img_ids) + len(val2_img_ids) == len(val_img_ids)
# how many examples have a 
def count_questions_with_diff_ans(ixs):
    N = len(ixs)
    count = 0
    for ix in ixs:
        ques = questions[ix]
        if len(ques['ques_with_diff_ans']) > 0:
            count += 1
    return f'{count} / {N} = {count / N}'
assert len(val1_img_ids.intersection(val2_img_ids)) == 0
print(f'train diff ans questions: {count_questions_with_diff_ans(train_ixs)}')
print(f'val1 diff ans questions: {count_questions_with_diff_ans(val1_ixs)}')
print(f'val2 diff ans questions: {count_questions_with_diff_ans(val2_ixs)}')

# convert ixs to image_ids
train_img_ids = sorted(list(set([questions[ix]['image_id'] for ix in train_ixs])))
print(f'train: {len(train_img_ids)}')
val1_img_ids = sorted(list(val1_img_ids))
print(f'val1: {len(val1_img_ids)}')
val2_img_ids = sorted(list(val2_img_ids))
print(f'val2: {len(val2_img_ids)}')

# save splits to file
out_name = pth.join(data_dir, 'split_info_v3.json')
with open(out_name, 'w') as f:
    json.dump({
        'splits': ['train', 'val1', 'val2'],
        'split_img_ids': {
            'train': train_img_ids,
            'val1': val1_img_ids,
            'val2': val2_img_ids,
        }
    }, f)

import json
import pdb
import nltk
from nltk.tokenize import word_tokenize
import re
import numpy as np
import h5py

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def build_vocab(vqa_ques, vqa_anno, imgs, build=True):

    count_thr = 5
    # count up the number of words
    counts = {}
    counts_pos = {}
    count = 0
    for question, annotation, img in zip(vqa_ques['questions'], vqa_anno['annotations'], imgs):
        ques = question['question'].lower()
        ans = annotation['multiple_choice_answer'].lower()
        ques_tokens = word_tokenize(ques)
        ques_pos = nltk.pos_tag(ques_tokens)
        ans_tokens = word_tokenize(ans)
        annotation['tokens'] = ques_tokens
        annotation['ans_tokens'] = ans_tokens
        img['image_id'] = annotation['image_id']
        img['question_id'] = annotation['question_id']
        for w in ques_tokens:
            counts[w] = counts.get(w, 0) + 1
        for w in ans_tokens:
            counts[w] = counts.get(w, 0) + 1
        count += 1
        if count % 10000 == 0: print(count)

    vocab = None
    vocab_pos = None

    if build == True:
        cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
        print('top words and their counts:')
        print('\n'.join(map(str,cw[:20])))

        # print some stats
        total_words = sum(counts.values())
        print('total words:', total_words)
        bad_words = [w for w,n in counts.items() if n <= count_thr]
        vocab = [w for w,n in counts.items() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
        print('number of words in vocab would be %d' % (len(vocab), ))
        print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

        cw_pos = sorted([(count,w) for w,count in counts_pos.items()], reverse=True)
        print('top words in pos and their counts:')
        print('\n'.join(map(str,cw_pos[:20])))
        vocab_pos = [w for count, w in cw_pos]
    
    return imgs, vocab, counts_pos


def build_ques2ans(vqa_ques, vqa_anno, imgs):

    ques2ans = {}
    for question, annotation in zip(vqa_ques['questions'], vqa_anno['annotations']):
        ques = question['question'].lower()
        ans = annotation['multiple_choice_answer'].lower()      
        ques_id = question['question_id']
        
        if ques not in ques2ans:
            ques2ans[ques] = []

        ques2ans[ques].append([ans, ques_id])

    # ques_with_same_ans = []
    # ques_with_diff_ans = []

    for question, annotation, img in zip(vqa_ques['questions'], vqa_anno['annotations'], imgs):
        ques = question['question'].lower()
        ans = annotation['multiple_choice_answer'].lower()      
        ques_id = question['question_id']

        same_tmp = []
        diff_tmp = []
        for possible_ans in ques2ans[ques]:
            if possible_ans[0] == ans:
                same_tmp.append(possible_ans[1])
            else:
                diff_tmp.append(possible_ans[1])

        img['ques_with_same_ans'] = same_tmp
        img['ques_with_diff_ans'] = diff_tmp

    return imgs

def create_data_mats(vqa_anno, vocab):

    max_ques_len = 20
    max_ans_len = 3
    num = len(vqa_anno)

    ques = np.zeros([num, max_ques_len])
    ans = np.zeros([num, max_ans_len])
    ques_len = np.zeros(num)
    ans_len = np.zeros(num)

    word2ind = {w:i+1 for i, w in enumerate(vocab)}
    UNK_ind = word2ind['<UNK>']

    for i, img in enumerate(vqa_anno):
        ques_tokens = img['tokens']
        ans_tokens = img['ans_tokens']

        ques_tokens = ['<START>'] + ques_tokens + ['<END>']

        for j, token in enumerate(ques_tokens):
            if j < max_ques_len:
                ques[i, j] = word2ind.get(token, UNK_ind)
        
        ques_len[i] = min(len(ques_tokens), max_ques_len)       

        for j, token in enumerate(ans_tokens):
            if j < max_ans_len:
                ans[i,j] = word2ind.get(token, UNK_ind)
        ans_len[i] = min(len(ans_tokens), max_ans_len)

    return ques, ques_len, ans, ans_len

if __name__ == "__main__":

    VQA_QUES_TRAIN_PATH = 'VQA/v2_OpenEnded_mscoco_train2014_questions.json'
    VQA_ANNO_TRAIN_PATH = 'VQA/v2_mscoco_train2014_annotations.json'
    VQA_QUES_VAL_PATH = 'VQA/v2_OpenEnded_mscoco_val2014_questions.json'
    VQA_ANNO_VAL_PATH = 'VQA/v2_mscoco_val2014_annotations.json'

    vqa_ques_train = json.load(open(VQA_QUES_TRAIN_PATH, 'r'))
    vqa_anno_train = json.load(open(VQA_ANNO_TRAIN_PATH, 'r'))
    vqa_ques_val = json.load(open(VQA_QUES_VAL_PATH, 'r'))
    vqa_anno_val = json.load(open(VQA_ANNO_VAL_PATH, 'r'))

    imgs_train = [{'split':'train'} for _ in vqa_ques_train['questions']]
    imgs_val = [{'split':'val'} for _ in vqa_ques_val['questions']]

    imgs_train, vocab, pos_vocab = build_vocab(vqa_ques_train, vqa_anno_train, imgs_train, True)
    imgs_val, _, _ = build_vocab(vqa_ques_val, vqa_anno_val, imgs_val, False)

    vocab.append('<UNK>')
    vocab.append('<START>')
    vocab.append('<END>')

    # same question with same answer. Those information helps on sample the pool information    
    imgs_train = build_ques2ans(vqa_ques_train, vqa_anno_train, imgs_train)
    imgs_val = build_ques2ans(vqa_ques_val, vqa_anno_val, imgs_val)

    # get the unique image.
    unique_image = {}
    for annotation in vqa_anno_train['annotations']:
        image_id = annotation['image_id']
        if image_id not in unique_image:
            unique_image[image_id] = len(unique_image)

    for annotation in vqa_anno_val['annotations']:
        image_id = annotation['image_id']
        if image_id not in unique_image:
            unique_image[image_id] = len(unique_image)

    imgs = imgs_train + imgs_val
    vqa_anno = vqa_anno_train['annotations'] + vqa_anno_val['annotations']
    ques, ques_len, ans, ans_len = create_data_mats(vqa_anno, vocab)

    # json.dump(imgs, open('v2_vqa_question.json', 'w'))
    json.dump({'imgs':imgs, 'unique_image':unique_image, 'vocab':vocab, 'pos_vocab':pos_vocab}, open('v2_vqa_info.json', 'w'))
    
    f = h5py.File('v2_vqa_data.h5', 'w')
    f.create_dataset('ques', dtype='uint32', data=ques)
    f.create_dataset('ques_len', dtype='uint32', data=ques_len)
    f.create_dataset('ans', dtype='uint32', data=ans)
    f.create_dataset('ans_len', dtype='uint32', data=ans_len)
    f.close()

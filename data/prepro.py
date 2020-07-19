import json
import pdb
import nltk
from nltk.tokenize import word_tokenize
import re
import numpy as np
import h5py


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


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
        ques_tokens = tokenize(ques)
        annotation['tokens'] = ques_tokens
        img['image_id'] = annotation['image_id']
        img['question_id'] = annotation['question_id']
        for w in ques_tokens:
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
    
    return imgs, vocab


def build_ques2ans(vqa_ques, vqa_anno, imgs):

    ques2ans = {}
    for question, annotation in zip(vqa_ques['questions'], vqa_anno['annotations']):
        ques = question['question'].lower()
        ans = annotation['multiple_choice_answer'].lower()      
        ques_id = question['question_id']
        
        if ques not in ques2ans:
            ques2ans[ques] = []

        ques2ans[ques].append([ans, ques_id])

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

def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence

def create_ans2label(occurence):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    return ans2label, label2ans

def compute_target(imgs, answers_dset, ans2label, name, cache_root='data/cache'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """

    target = []
    for ans_entry, img in zip(answers_dset, imgs):
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)
        
        img['labels'] = labels
        img['scores'] = scores

    return imgs


def create_data_mats(vqa_anno, vocab):

    max_ques_len = 20
    num = len(vqa_anno)

    ques = np.zeros([num, max_ques_len])
    ques_len = np.zeros(num)

    word2ind = {w:i+1 for i, w in enumerate(vocab)}
    UNK_ind = word2ind['<UNK>']

    for i, img in enumerate(vqa_anno):
        ques_tokens = img['tokens']

        ques_tokens = ['<START>'] + ques_tokens + ['<END>']

        for j, token in enumerate(ques_tokens):
            if j < max_ques_len:
                ques[i, j] = word2ind.get(token, UNK_ind)
        
        ques_len[i] = min(len(ques_tokens), max_ques_len)

    return ques, ques_len

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(val) for val in vals[1:]]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

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

    imgs_train, vocab = build_vocab(vqa_ques_train, vqa_anno_train, imgs_train, True)
    imgs_val, _ = build_vocab(vqa_ques_val, vqa_anno_val, imgs_val, False)

    vocab.append('<UNK>')
    vocab.append('<START>')
    vocab.append('<END>')

    # same question with same answer. Those information helps on sample the pool information    
    imgs_train = build_ques2ans(vqa_ques_train, vqa_anno_train, imgs_train)
    imgs_val = build_ques2ans(vqa_ques_val, vqa_anno_val, imgs_val)

    answers = vqa_anno_train['annotations'] + vqa_anno_val['annotations']
    # compute the softscore for the answers.
    occurence = filter_answers(answers, 9)
    ans2label, label2ans = create_ans2label(occurence)

    imgs_train = compute_target(imgs_train, vqa_anno_train['annotations'], ans2label, 'train')
    imgs_val = compute_target(imgs_val, vqa_anno_val['annotations'], ans2label, 'val')

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
    ques, ques_len = create_data_mats(vqa_anno, vocab)

    # json.dump(imgs, open('v2_vqa_question.json', 'w'))
    json.dump({'imgs':imgs, 'unique_image':unique_image, 'vocab':vocab, 'ans2label':ans2label}, open('v2_vqa_info.json', 'w'))
    
    f = h5py.File('v2_vqa_data.h5', 'w')
    f.create_dataset('ques', dtype='uint32', data=ques)
    f.create_dataset('ques_len', dtype='uint32', data=ques_len)
    f.close()

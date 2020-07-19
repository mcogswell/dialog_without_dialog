import json
import pdb
from stanfordnlp.server import CoreNLPClient
import numpy as np
from collections import defaultdict
from itertools import islice
pattern_use_counter = defaultdict(lambda: 0)

def extract_binary(client, ques, ann):
	patterns = []
	matches = []
	qtype = None
	sub = None
	openie = []
	tokens = ann.sentence[0].token
	offset = 0
	lemmas = [token.lemma for token in tokens]

	Flag = True
	if ',' in lemmas:
		pos = lemmas.index(',')
		if tokens[pos+1].lemma in ['be', 'have', 'do'] or tokens[pos+1].pos == 'MD':
			offset = pos + 1

	if tokens[offset].lemma in ['be', 'have', 'some'] or tokens[offset].pos == 'MD':
		# check the second word, if there is a det relationship.
		loc = None#1 + offset
		if tokens[offset+1].pos == 'DT':
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub < nsubj{tag:/(NN.*|VBG|JJ)/}=qtype > {lemma:%s}"%tokens[offset+1].lemma)
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub >/(compound|iobj|dobj|nmod:.*)/ {}=qtype > {lemma:%s}"%tokens[offset+1].lemma)
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub >amod{tag:/(NN.*|VB.*|JJ)/} = qtype > {lemma:%s}"%tokens[offset+1].lemma)
			# is the animal a mammal ?
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub >{tag:/(NN.*|VBG|JJ)/}=qtype >{lemma:%s}"%tokens[offset+1].lemma)
			# are any of the animals eating ?
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub >{tag:/(NN.*|VBG|JJ)/}=qtype < /nmod:.*/{lemma:%s}"%tokens[offset+1].lemma)
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub < /nmod:.*/{lemma:%s}"%tokens[offset+1].lemma)
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub > {lemma:%s}"%tokens[offset+1].lemma)
			patterns.append("{tag:/(NN.*|VBG|JJ)/}=sub")
			Flag = False

	if Flag:
		# use the subject of some verb if possible
		patterns.append('{pos:/NN.*|PR.*/}=sub <nsubj {pos:/VB.*/}')

		# otherwise, use the subject of a verb phrase like "is blue" as in "is the man blue"
		patterns.append('{pos:/NN.*|PR.*/}=sub </nsubj|amod/ ({pos:/NN.*|JJ.*/}=qtype >cop {pos:/VB.*/})')

		# worst case: just find any noun (still doesn't catch quite everything)
		patterns.append('{pos:/NN.*/}=sub')

	# find the word, and conver to caption.
	for pattern in patterns:
		match = client.semgrex(ques, pattern)
		if match['sentences'][0]['length'] != 0:
			break

	if match['sentences'][0]['length'] != 0:
		if '$qtype' in match['sentences'][0]['0']:
			qtype = 'is_' + tokens[match['sentences'][0]['0']['$qtype']['begin']].lemma
		if '$sub' in match['sentences'][0]['0']:
			sub = tokens[match['sentences'][0]['0']['$sub']['begin']].lemma

	for triple in ann.sentence[0].openieTriple:
		openie.append([triple.subject, triple.relation, triple.object])

	return qtype, sub, openie

def extract_others(client, ques, ann):
	patterns = []
	matches = []
	qtype = None
	sub = None
	openie = []
	tokens = ann.sentence[0].token

	# how many christmas tree he is decorating ?
	# what number can be seen ?
	# which platforms are the trains near ?

	patterns.append("{tag:/(NN.*|VB.*|JJ)/}=qtype >/(det|advmod)/{lemma:%s}"%tokens[0].lemma)
	# the target word is any noun word other than that.

	# what is this photo taken looking through?
	# what is the person doing?
 	# what is in the person's hand?
 	# what is the white streak?
 	# what is the business man doing in the picture?
	patterns.append("{tag:/(NN.*|VB.*|JJ)/}=sub > /(case|amod|acl)/ {}=qtype >nsubj {lemma:%s}"%tokens[0].lemma)
	patterns.append("{tag:/(NN.*|VB.*|JJ)/}=sub > /(case|amod|acl)/ {}=qtype <nsubj {lemma:%s}"%tokens[0].lemma)
	patterns.append("{tag:/(NN.*|VB.*|JJ)/}=qtype > nsubj {}=sub >dobj {lemma:%s}"%tokens[0].lemma)
	patterns.append("{tag:/(NN.*|VB.*|JJ)/}=sub <nsubj {lemma:%s}"%tokens[0].lemma)


	for pattern in patterns:
		match = client.semgrex(ques, pattern)
		if match['sentences'][0]['length'] != 0:
			break

	if match['sentences'][0]['length'] != 0:
		if '$qtype' in match['sentences'][0]['0']:
			qtype_tmp = tokens[match['sentences'][0]['0']['$qtype']['begin']].lemma
			qtype = tokens[0].lemma + '_' + qtype_tmp
		if '$sub' in match['sentences'][0]['0']:
			sub = tokens[match['sentences'][0]['0']['$sub']['begin']].lemma

		if sub == None:
		# if the sub is None, find the first NN that is not the qtype.
			for token in tokens:
				if token.word != qtype_tmp and token.pos in ['NN', 'NNS']:
					sub = token.word
					break

	for triple in ann.sentence[0].openieTriple:
		openie.append([triple.subject, triple.relation, triple.object])

	return qtype, sub, openie

def filter_question(ques):
	ques = ques.lower()
	return ques

VQA_QUES_PATH = 'VQA/v2_OpenEnded_mscoco_train2014_questions.json'
VQA_ANNO_PATH = 'VQA/v2_mscoco_train2014_annotations.json'

vqa_ques = json.load(open(VQA_QUES_PATH, 'r'))
vqa_anno = json.load(open(VQA_ANNO_PATH, 'r'))

# first we extract the attribute words. The attribute words is associate with 
count = 0
with CoreNLPClient(annotators=['tokenize','lemma','depparse', 'openie'], timeout=30000, memory='16G') as client:
	for question, annotation in zip(vqa_ques['questions'], vqa_anno['annotations']):
		ques = question['question']
		ans = annotation['multiple_choice_answer']
		ques_id = question['question_id']
		ques = ques.lower()
		ann =client.annotate(ques)
		if ans in ['yes', 'no']:
			# pass
			qtype, sub, openie = extract_binary(client, ques, ann)
			# binary_file.write('%d\t%s\n'%(ques_id, new_cap)) 
		else:
			qtype, sub, openie = extract_others(client, ques, ann)

		count += 1

		# save result in json

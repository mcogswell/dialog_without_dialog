import json
import pdb
from stanfordnlp.server import CoreNLPClient

from collections import defaultdict
from itertools import islice

pattern_use_counter = defaultdict(lambda: 0)

def extract_is(client, ques, tokens):
	patterns = []
	matches = []

	sub = None
	qtype = 'is'

	# This fixes some problems like the following:
	#   this provides "does his tie pair well with his suit?" -> tie
	#   instead of
	#   "does his tie pair well with his suit?" -> pair
	# But it also fails often
	#patterns.append('{pos:/NN.*|PR.*/}=sub <compound ({} <nsubj {pos:/VB.*/})')

	# use the subject of some verb if possible
	patterns.append('{pos:/NN.*|PR.*/}=sub <nsubj {pos:/VB.*/}')

	# otherwise, use the subject of a verb phrase like "is blue" as in "is the man blue"
	patterns.append('{pos:/NN.*|PR.*/}=sub </nsubj|amod/ ({pos:/NN.*|JJ.*/}=qtype >cop {pos:/VB.*/})')

	# worst case: just find any noun (still doesn't catch quite everything)
	patterns.append('{pos:/NN.*/}=sub')


	for ip, pattern in enumerate(patterns):
		match = client.semgrex(ques, pattern)
		if match['sentences'][0]['length'] > 0:
			#print(pattern)
			pattern_use_counter[pattern] += 1
			matches.append(match)
			break
	else:
	    print('unmatched: ', ques)
	    return qtype, sub

	if match['sentences'][0]['length'] != 0:
		if '$qtype' in match['sentences'][0]['0']:
			qtype = 'is ' + match['sentences'][0]['0']['$qtype']['text']
		if '$sub' in match['sentences'][0]['0']:
			sub = tokens[match['sentences'][0]['0']['$sub']['begin']].lemma

	return qtype, sub


def extract_how_many(client, ques, tokens):
	sub = None
	qtype = None

	patterns = []
	matches = []
	if tokens[0].word == 'how':
		# how many christmas tree he is decorating?
		patterns.append('{}=qtype >advmod {lemma:how} < {}=sub')
		# how to the boys not lose their boards ?
		patterns.append('{}=qtype >advmod {lemma:how} >/nmod:.*/ {}=sub')
		# how color peppers are on the plate?
		patterns.append('{}=sub >> {lemma:how}')

	elif tokens[0].word == 'what':
		# what number is on the front of the train ?
		patterns.append('{} >nsubj ({}=qtype >det {lemma:what}) >/(iobj|dobj|nmod:.*)/ {tag:/NNP?S?/}=sub')

		# what is the number on the front of the train?
		patterns.append('{}=qtype <nsubj {lemma:what} > /(iobj|dobj|nmod:.*)/{}=sub')

		# what number can be seen ?
		patterns.append('{}=qtype >det {lemma:what}')

		# what number of dogs are being walked ?
		patterns.append('{}=qtype >det {lemma:what} >/(iobj|dobj|nmod:.*)/ {tag:/NNP?S?/}=sub')


	for pattern in patterns:
		match = client.semgrex(ques, pattern)
		if match['sentences'][0]['length'] > 0:
			matches.append(match)
			break

	if match['sentences'][0]['length'] != 0:
		if '$qtype' in match['sentences'][0]['0']:
			qtype = match['sentences'][0]['0']['$qtype']['text']
		if '$sub' in match['sentences'][0]['0']:
			sub = tokens[match['sentences'][0]['0']['$sub']['begin']].lemma
	else:
		pdb.set_trace()



	# if tokens[0].word == 'how' and tokens[1].word == 'many':
	# 	# pattern = '{tag:/NN.*/}=sub > amod {lemma:many;tag:JJ}'
	# 	# result = client.semgrex(ques, pattern)
	# 	# qtype = 'many'
	# 	# sub = tokens[result['sentences'][0]['0']['$sub']['begin']].lemma

	# 	# if sub in ['kinds', 'types', 'kind', 'type']:
	# 	# 	pattern = '{tag:/NN.*/} > amod {lemma:many;tag:JJ} >/nmod:.*/ {tag:/NN.*/}=sub'
	# 	# 	result = client.semgrex(ques, pattern)
	# 	# 	qtype = 'kind'
	# 	# 	sub = tokens[result['sentences'][0]['0']['$sub']['begin']].lemma

	# # 	qtype = 'many'
	# # 	for token in tokens:
	# # 		if token.pos in ['NN', 'NNS']:
	# # 			if token.lemma in ['kinds', 'types', 'kind', 'type']:
	# # 				continue
	# # 			sub = token.lemma
	# # 			break
	# # 	if sub == None:
	# # 		for token in tokens:
	# # 			if token.pos in ['VBG', 'VB', 'VBN']:
	# # 				sub = token.lemma
	# # 				break				
	# # elif tokens[0].word == 'how':
	# # 	pattern = '{}=sub > advmod {tag:RB} = qtype'
	# # 	result = client.semgrex(ques, pattern)
		
	# # 	if result['sentences'][0]['length'] != 0:
	# # 		qtype = result['sentences'][0]['0']['$qtype']['text']
	# # 		sub = tokens[result['sentences'][0]['0']['$sub']['begin']].lemma
	# # 	else:
	# # 		for token in tokens:
	# # 			if token.pos in ['NN', 'NNS']:
	# # 				sub = token.lemma
	# # 				qtype = 'many'
	# # elif 'what number' in ques or 'what is the number' in ques:
	# # 	qtype = 'number'
	# # 	for token in tokens:
	# # 		if token.pos in ['NN', 'NNS'] and token.lemma != 'number':
	# # 			sub = token.lemma
	# # 			# qtype = 'number'
	# # 			break

	# # elif tokens[0].word in ['is', 'are']:
	# # 	qtype = 'number_choice'
	# # 	for token in tokens:
	# # 		if token.pos in ['NN', 'NNS']:
	# # 			sub = token.lemma
	# # 			# qtype = 'number'
	# # 			break
	# # if ques
	return qtype, sub


def filter_question(ques):
	ques = ques.lower()

	ques = ques.split(',')[-1]
	# it look like --> the
	ques = ques.replace('it look like', '') 
	# do you think --> ''
	ques = ques.replace('do you think', '')
	# can you tell --> what is/are
	#			   --> "" is there is what and how
	ques = ques.replace('can you tell', '')
	# would you say --> is
	ques = ques.replace('would you say', '')
	# that you can see --> ""
	ques = ques.replace('that you can see', '')
	# do you see --> is 
	ques = ques.replace('do you see', '')

	ques = ques.replace('can you see', '')
	# would you guess
	ques = ques.replace('would you guess', '')
	# can you see --> is there
	ques = ques.replace('can you see', '')

	return ques

VQA_QUES_PATH = 'VQA/v2_OpenEnded_mscoco_train2014_questions.json'
VQA_ANNO_PATH = 'VQA/v2_mscoco_train2014_annotations.json'

vqa_ques = json.load(open(VQA_QUES_PATH, 'r'))
vqa_anno = json.load(open(VQA_ANNO_PATH, 'r'))

counts = {}
ans2ques = {}
for question, annotation in zip(vqa_ques['questions'], vqa_anno['annotations']):
	answer = annotation['multiple_choice_answer']
	ques = question['question']
	counts[answer] = counts.get(answer, 0) + 1
	if answer not in ans2ques:
		ans2ques[answer] = {}
	ans2ques[answer][ques] = ans2ques[answer].get(ques,0) + 1

cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
print('top words and their counts:')
print('\n'.join(map(str,cw[:20])))

# print some stats
total_ans = sum(counts.values())
print('total answers:', total_ans)
ans_vocab = [ans[1] for ans in cw[:]]
ans_ques = [ans2ques[ans] for ans in ans_vocab]

with CoreNLPClient(annotators=['tokenize','lemma','depparse'], timeout=30000, memory='16G') as client:
	for i, questions in enumerate(ans_ques):
		ans = ans_vocab[i]
		for ques, count in questions.items():
			ques = filter_question(ques)
			ann =client.annotate(ques)
			tokens = ann.sentence[0].token
			print(ques)
			if ans in ['yes', 'no']:
				qtype, sub = extract_is(client, ques, tokens)
			elif ans.isdigit():
				qtype, sub = extract_how_many(client, ques, tokens)
			print(qtype, sub)

		#ques = questions['question']
		# ann =client.annotate(ques)

		print()
		pdb.set_trace()
	# pdb.set_trace()

	# if ques in ques2type:
	# 	ques_type = ques2type[ques]
	# 	question['ques_type'] = ques_type
	# else:
	# 	print(ques)

print('total yes: ', len(ans_ques[0]))
print('total no: ', len(ans_ques[1]))
print('total: ', len(ans_ques[0]) + len(ans_ques[1]))
import pprint
pprint.pprint(pattern_use_counter)

# json.dump(vqa_ques, open(VQA_QUES_PATH, 'w'))


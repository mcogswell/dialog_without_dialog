import os
import os.path as pth
import json
import argparse
import pickle as pkl
import random
import itertools
import datetime
import math
import hashlib
from subprocess import Popen

import pprint
import tabulate

import boto3
import joblib
import jinja2
import xmltodict

parser = argparse.ArgumentParser(
            description='MTurk scripting to manage HITs that compare a set of models.')
parser.add_argument('command', choices=['generate_hits', 'launch_hits',
                                        'retrieve', 'status', 'delete_hits',
                                        'approve'])
parser.add_argument('savePath')
parser.add_argument('-expCodes',
                    help='Comma separated list of exp codes for the models to compare')
parser.add_argument('-mturkEnv', choices=['sandbox', 'production'],
                                 default='sandbox')
parser.add_argument('-numExamples', default=51, type=int)
parser.add_argument('-hitType', choices=['guessing_game_v1', 
                                         'question_comparison_v1',
                                         'question_comparison_v2'],
                                default='guessing_game_v1')
parser.add_argument('-visCode', default='eval3.{batch_index}.2.1.3')


region_name = 'us-east-1'


def get_assignment_fname(hits_dir, hit, assignment):
    assert assignment is not 'NA'
    os.makedirs(pth.join(hits_dir, 'results'), exist_ok=True)
    fname = pth.join(hits_dir, 'results', f"HITTypeId{hit['HITTypeId']}_HITID{hit['HITId']}_ASNID{assignment['AssignmentId']}.pkl")
    return fname


def get_num_hits(args):
    exp_codes = args.expCodes.split(',')
    if args.hitType == 'guessing_game_v1':
        instances_per_hit = 1
    elif args.hitType == 'question_comparison_v1':
        instances_per_hit = 5
    elif args.hitType == 'question_comparison_v2':
        instances_per_hit = 5
    fac = math.factorial
    binom = lambda n, k: fac(n) // (fac(k) * fac(n - k))
    num_pairs = binom(len(exp_codes), 2)
    num_hits = num_pairs * args.numExamples // instances_per_hit
    assert args.numExamples % instances_per_hit == 0
    return num_hits, instances_per_hit


def parse_answer(answer):
    result = {}
    for d in xmltodict.parse(answer)['QuestionFormAnswers']['Answer']:
        val = d['FreeText']
        try:
            val = int(val)
        except (ValueError, TypeError) as e:
            pass
        result[d['QuestionIdentifier']] = val
    return result


def pretify_question(question):
    # capitalize
    question = question[0].capitalize() + question[1:]
    # adjust punctuation spacing
    question = question.replace(" '", "'")
    question = question.replace("' s", "'s")
    question = question.replace(" ?", "?")
    question = question.replace(" ( s )", "(s)")
    return question


def generate_dialog_data(args, batch_index):
    exp_codes = args.expCodes.split(',')
    dialog_data = {}
    for exp_code in exp_codes:
        # TODO: parameterize data/experiments
        data_fname = pth.join('data/experiments', exp_code, 'mturk', args.visCode.format(batch_index=batch_index) + '_dialog_data.joblib')
        if not pth.exists(data_fname):
            cmd = f'python run.py -m visualize -p debug_noslurm {exp_code} {args.visCode.format(batch_index=batch_index)}'
            print(f'generating file {data_fname} using command')
            print(cmd)
            print('<', '-'*79)
            Popen(cmd, shell=True).wait()
            print('-'*79, '>')
        else:
            print(f'using existing data {data_fname}')
        dialog_data[exp_code] = data_fname
    return dialog_data


def find_matching_examples(dialog_data, batch_index):
    exp_codes = list(dialog_data.keys())
    nexamples = len(dialog_data[exp_codes[0]]['data_by_batch'][batch_index]['examples'])
    exi = 0
    matching_idxs = []
    for exi in range(nexamples):
        targets = [dialog_data[exp_code]['data_by_batch'][batch_index]['examples'][exi]['target'] for exp_code in exp_codes]
        assert len(set(targets)) == 1
        target = targets[0]
        rounds = [dialog_data[exp_code]['data_by_batch'][batch_index]['rounds'] for exp_code in exp_codes]
        assert len(set(map(len, rounds))) == 1, 'Not all dialogs have the same number of rounds'
        # final round predictions per model
        preds = [dialog_data[exp_code]['data_by_batch'][batch_index]['rounds'][-1]['preds'][exi] for exp_code in exp_codes]
        if all(pred == target for pred in preds):
            matching_idxs.append(exi)
    print(f'{len(matching_idxs)} / {nexamples} matching examples in batch {batch_index}')
    return matching_idxs


hit_types = {
    'guessing_game_v1': {
        'template': 'templates/mturk_guessing_game_performance.html',
        'hit_type_args': {
            'Title': 'Which image are the two bots talking about?',
            'Description': "Two bots have a dialog about some images. Your job is to "
                        "guess which image they're talking about.",
            'Keywords': 'fast,ai,bots,dialog,images,guesswhich',
            'Reward': '0.05',
            'AssignmentDurationInSeconds': 600, # 10 minutes
            'AutoApprovalDelayInSeconds': 14400, # 4 hours
        },
    },
    'question_comparison_v1': {
        'template': 'templates/mturk_question_comparison.html',
        'hit_type_args': {
            'Title': 'Which question is more fluent English?',
            'Description': "Two bots asked questions about some images. "
                        "Which question uses better English?",
            'Keywords': 'fast,ai,bots,language comparison,questions',
            'Reward': '0.10',
            'AssignmentDurationInSeconds': 600, # 10 minutes
            'AutoApprovalDelayInSeconds': 14400, # 4 hours
        },
    },
    'question_comparison_v2': {
        'template': 'templates/mturk_question_comparison.html',
        'hit_type_args': {
            'Title': 'Which question focuses on the images?',
            'Description': "Two bots asked questions about some images. "
                        "Which question uses better English?",
            'Keywords': 'fast,ai,bots,language comparison,questions',
            'Reward': '0.15',
            'AssignmentDurationInSeconds': 600, # 10 minutes
            'AutoApprovalDelayInSeconds': 14400, # 4 hours
        },
    },
}


def generate_hits_game(args, hits_dir, examples_per_hit, max_batches=1000):
    exp_codes = args.expCodes.split(',')
    dialog_data = {exp_code: {'data_by_batch': {}} for exp_code in exp_codes}
    num_models = len(exp_codes)
    assert args.numExamples % num_models == 0, 'important so there are the same number of examples per model'

    # find examples where all models have the correct response at the final round
    matching_example_idxs = []
    batch_index = 0
    for batch_index in range(max_batches):
        for exp_code, fname in generate_dialog_data(args, batch_index).items():
            dialog_data[exp_code]['data_by_batch'][batch_index] = joblib.load(fname)
        matching = find_matching_examples(dialog_data, batch_index)
        matching_example_idxs.extend((batch_index, exi) for exi in matching)
        if len(matching_example_idxs) >= args.numExamples:
            matching_example_idxs = matching_example_idxs[:args.numExamples]
            break
    else:
        raise Exception(f'Only found {len(matching_example_idxs)} examples '
                        f'where all models got it right in {max_batches} batches.')

    # generate data for individual hits
    hit_data = []
    all_pools = [] # just to track uniqueness
    for mei, tup in enumerate(matching_example_idxs):
        batch_index, exi = tup
        # use one model per pool, going to the next each time
        exp_code = exp_codes[mei % num_models]
        hit_id = mei // examples_per_hit
        new_hit = (mei % examples_per_hit == 0)
        d = dialog_data[exp_code]['data_by_batch'][batch_index]
        if new_hit:
            hit = {
                'examples': [d['examples'][exi]],
                'rounds': [{k: rnd[k][exi:exi+1] for k in rnd} for rnd in d['rounds']],
                'wrap_period': 2,
            }
            hit_data.append(hit)
        else:
            hit['examples'].append(d['examples'][exi])
            for rndi, rnd in enumerate(hit['rounds']):
                for k in rnd:
                    rnd[k].append(d['rounds'][rndi][k][exi])
        # TODO: pretify these questions
        hit['examples'][-1]['batch_index'] = batch_index
        hit['examples'][-1]['exi'] = exi
        hit['examples'][-1]['exp_code'] = exp_code
        hit['examples'][-1]['eval_code'] = args.visCode.format(batch_index=batch_index)
        all_pools.append(tuple(d['examples'][exi]['img_urls']) + (d['examples'][exi]['target'],))
    assert len(all_pools) == len(set(all_pools)), 'pool images and targets should be unique'


    # render HIT to html and xml docs
    with open('templates/mturk_guessing_game_performance.html', 'r') as f:
        html_template = jinja2.Template(f.read())
    with open('templates/mturk_hit_template.xml', 'r') as f:
        hit_template = f.read()
    hit_xml_paths = []
    for hiti, hit in enumerate(hit_data):
        hit_html = html_template.render(hit)
        hit_xml = hit_template.replace('{{html}}', hit_html)

        # save docs it to the exp/eval directory
        hit_html_fname = pth.join(hits_dir, f'hit_{hiti}.html')
        with open(hit_html_fname, 'w') as f:
            f.write(hit_html)
        hit_fname = pth.join(hits_dir, f'hit_{hiti}.xml')
        with open(hit_fname, 'w') as f:
            f.write(hit_xml)
        print(f'wrote HIT question file to {hit_fname}')
        hit_xml_paths.append(hit_fname)
    return hit_xml_paths


def generate_hits_question_comparison(args, hits_dir, examples_per_hit, max_batches=1000):
    exp_codes = args.expCodes.split(',')
    dialog_data = {exp_code: {'data_by_batch': {}} for exp_code in exp_codes}
    num_models = len(exp_codes)

    # find examples where all models have the correct response at the final round
    matching_example_idxs = []
    batch_index = 0
    for batch_index in range(max_batches):
        for exp_code, fname in generate_dialog_data(args, batch_index).items():
            dialog_data[exp_code]['data_by_batch'][batch_index] = joblib.load(fname)
        matching = find_matching_examples(dialog_data, batch_index)
        matching_example_idxs.extend((batch_index, exi) for exi in matching)
        if len(matching_example_idxs) >= args.numExamples:
            matching_example_idxs = matching_example_idxs[:args.numExamples]
            break
    else:
        raise Exception(f'Only found {len(matching_example_idxs)} examples '
                        f'where all models got it right in {max_batches} batches.')

    # generate the data for each (pool, model pair)
    all_pairs = [] # just to track uniqueness
    pair_data = []
    for tup in matching_example_idxs:
        batch_index, exi = tup
        # use one model per pool, going to the next each time
        num_rounds = len(dialog_data[exp_codes[0]]['data_by_batch'][batch_index]['rounds'])
        # Use the same seed for each pool and hit type. That way different
        # hit types with the same set of examples are randomized differently,
        # but the code produces exactly the same hits from one run to the next.
        random.seed(exi + int(hashlib.sha1(args.hitType.encode()).hexdigest(), 16))
        roundi = random.randrange(num_rounds)
        # this is the total list of exp_codes, but present them one pair at a time
        _all_exp_codes = random.sample(exp_codes, k=len(exp_codes))
        model_pairs = list(itertools.combinations(_all_exp_codes, r=2))
        for _exp_codes in model_pairs:
            round_by_models = []
            for exp_code in _exp_codes:
                d = dialog_data[exp_code]['data_by_batch'][batch_index]
                rnd = d['rounds'][roundi]
                rnd['questions'] = [pretify_question(q) for q in rnd['questions']]
                round_by_models.append({k: rnd[k][exi] for k in rnd})
            pair_data.append({
                'exi': exi,
                'roundi': roundi,
                'batch_index': batch_index,
                'example': d['examples'][exi],
                'rounds_by_model': round_by_models,
                'exp_codes': _exp_codes,
                'eval_code': args.visCode.format(batch_index=batch_index),
            })
            all_pairs.append(tuple(d['examples'][exi]['img_urls']) + (d['examples'][exi]['target'],) + _exp_codes)
        assert len(all_pairs) == len(set(all_pairs)), '(image, target, model1, model2) tuples should be unique'

    # Shuffle pair data and generate one datum per HIT. Shuffle differently
    # for different sets of examples, but the same across runs for the same
    # set of examples.
    random.seed(len(pair_data))
    random.shuffle(pair_data)
    examples_per_hit = 5
    hit_data = []
    for pairi, pair in enumerate(pair_data):
        hit_id = pairi // examples_per_hit
        new_hit = (pairi % examples_per_hit == 0)
        if new_hit:
            hit = {
                'pair_data': [],
                'wrap_period': 2,
            }
            hit_data.append(hit)
        hit['pair_data'].append(pair)

    # render HIT to html and xml docs
    if args.hitType == 'question_comparison_v1':
        template_fname = 'templates/mturk_question_comparison.html'
    elif args.hitType == 'question_comparison_v2':
        template_fname = 'templates/mturk_question_comparison_grounded.html'
    with open(template_fname, 'r') as f:
        html_template = jinja2.Template(f.read())
    with open('templates/mturk_hit_template.xml', 'r') as f:
        hit_template = f.read()
    hit_xml_paths = []
    for hiti, hit in enumerate(hit_data):
        hit_html = html_template.render(hit)
        hit_xml = hit_template.replace('{{html}}', hit_html)

        # save docs it to the exp/eval directory
        hit_html_fname = pth.join(hits_dir, f'hit_{hiti}.html')
        with open(hit_html_fname, 'w') as f:
            f.write(hit_html)
        hit_fname = pth.join(hits_dir, f'hit_{hiti}.xml')
        with open(hit_fname, 'w') as f:
            f.write(hit_xml)
        print(f'wrote HIT question file to {hit_fname}')
        hit_xml_paths.append(hit_fname)
    return hit_xml_paths


def generate_hits(args, hits_dir, examples_per_hit):
    if args.hitType == 'guessing_game_v1':
        return generate_hits_game(args, hits_dir, examples_per_hit)
    elif args.hitType in ['question_comparison_v1', 'question_comparison_v2']:
        return generate_hits_question_comparison(args, hits_dir, examples_per_hit)


def log_hit(hits_dir, hit):
    hit_info_fname = pth.join(hits_dir, 'hit_info.json')
    if pth.exists(hit_info_fname):
        with open(hit_info_fname, 'r') as f:
            hit_info = json.load(f)
    else:
        hit_info = {'hits': []}
    hit_info['hits'].append(hit)
    # overwrite anything that exists already, having initialized from this file
    with open(hit_info_fname, 'w') as f:
        json.dump(hit_info, f)


def create_hit(args, mturk, hits_dir, hit_xml_paths, max_assignments):
    # get HIT type
    hit_type_args = hit_types[args.hitType]['hit_type_args']
    hit_type = mturk.create_hit_type(**hit_type_args)

    # create HITs on mturk
    for hit_internal_id, hit_xml_path in enumerate(hit_xml_paths):
        with open(hit_xml_path, 'r') as f:
            hit_xml = f.read()
        # create HIT
        new_hit = mturk.create_hit_with_hit_type(
            HITTypeId=hit_type['HITTypeId'],
            MaxAssignments=max_assignments,
            LifetimeInSeconds=172800, # 2 days
            Question=hit_xml,
        )

        # save HIT info
        log_hit(hits_dir, {
            'env': args.mturkEnv,
            'hit_internal_id': hit_internal_id,
            'hit_xml_path': hit_xml_path,
            'HITGroupId': new_hit['HIT']['HITGroupId'],
            'HITId': new_hit['HIT']['HITId'],
            'HITTypeId': hit_type['HITTypeId'],
        })

    return new_hit, hit_type


def get_assignments(args, mturk, hits_dir, include_disposed=False):
    try:
        with open(pth.join(hits_dir, 'hit_info.json'), 'r') as f:
            hit_info = json.load(f)
    except FileNotFoundError:
        print(f'No HITs for {args.expCode} {args.visCode}')
        return

    for _hiti, _hit in enumerate(hit_info['hits']):
        hit = mturk.get_hit(HITId=_hit['HITId'])['HIT']
        response = mturk.list_assignments_for_hit(HITId=hit['HITId'])
        if hit['HITStatus'] == 'Disposed' and not include_disposed:
            continue
        yield hit, response['Assignments']


def show_status(args, mturk, hits_dir, hit_base_url):
    rows = []
    for hit, assignments in get_assignments(args, mturk, hits_dir, include_disposed=True):
        if len(assignments) == 0:
            assignments = ['NA']
        for assignment in assignments:
            rows.append([
                hit['HITId'],
                hit['HITStatus'],
                hit['Reward'],
                'NA' if assignment == 'NA' else assignment['AssignmentId'],
                'NA' if assignment == 'NA' else assignment['WorkerId'],
                'NA' if assignment == 'NA' else assignment['AssignmentStatus'],
                'NA' if assignment == 'NA' else pth.exists(get_assignment_fname(hits_dir, hit, assignment)),
                hit_base_url + hit['HITGroupId'],
            ])

    print(tabulate.tabulate(rows,
        headers=['HIT Id', 'HIT Status', 'Reward', 'Assignment Id', 'Worker Id', 'Assignment Status', 'Assignment Retrieved', 'HIT URL']))


def retrieve_results(args, mturk, hits_dir):
    for hit, assignments in get_assignments(args, mturk, hits_dir):
        for assignment in assignments:
            fname = get_assignment_fname(hits_dir, hit, assignment)
            with open(fname, 'wb') as f:
                pkl.dump({'hit': hit, 'assignment': assignment}, f)


def delete_hits(args, mturk, hits_dir):
    for hit, assignments in get_assignments(args, mturk, hits_dir):
        if hit['HITStatus'] in ['Assignable', 'Unassignable']:
            print(f'Expiring HIT {hit["HITId"]}')
            mturk.update_expiration_for_hit(
                HITId=hit['HITId'],
                ExpireAt=datetime.datetime.now(),
            )
        if len(assignments) == 0:
            assignments = ['NA']
        ready = True
        for assignment in assignments:
            if assignment == 'NA':
                continue
            fname = get_assignment_fname(hits_dir, hit, assignment)
            if not pth.exists(fname):
                print(f'Assignment results for ASNID {assignment["AssignmentId"]} still need to be retrieved.')
                ready = False
            if assignment['AssignmentStatus'] != 'Approved':
                print(f'Assignment ASNID {assignment["AssignmentId"]} still need to be approved.')
                ready = False
        if ready:
            print(f'Deleting HIT {hit["HITId"]} of type {hit["HITTypeId"]}')
            mturk.delete_hit(
                HITId=hit['HITId'],
            )


def approve_assignments(args, mturk, hits_dir):
    for hit, assignments in get_assignments(args, mturk, hits_dir):
        if len(assignments) == 0:
            assignments = ['NA']
        ready = True
        for assignment in assignments:
            if assignment == 'NA' or assignment['AssignmentStatus'] != 'Submitted':
                continue
            print(f'Approving assignment {assignment["AssignmentId"]}')
            mturk.approve_assignment(
                AssignmentId=assignment['AssignmentId'],
                RequesterFeedback='Thanks for doing our HIT!',
            )


def main(args):
    # this should be a dict with two keys, 'access_key' and 'secret_key'
    with open('.mturk_keys.json', 'r') as f:
        safe = json.load(f)
        aws_access_key_id = safe['access_key']
        aws_secret_access_key = safe['secret_key']

    mturk_env = args.mturkEnv
    endpoint_url = {
        'sandbox': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com',
        'production': 'https://mturk-requester.us-east-1.amazonaws.com',
    }[mturk_env]
    hit_base_url = {
        'sandbox': 'https://workersandbox.mturk.com/mturk/preview?groupId=',
        'production': 'https://worker.mturk.com/mturk/preview?groupId=',
    }[mturk_env]
    max_assignments = {
        'sandbox': 1,
        'production': 3,
    }[mturk_env]
    print(f'using endpoint: {endpoint_url}')
    print(f'using hit_base_url: {hit_base_url}')

    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    hits_dir = pth.join(args.savePath, 'hits')
    os.makedirs(hits_dir, exist_ok=True)

    # This will return $10,000.00 in the MTurk Developer Sandbox
    #print(mturk.get_account_balance()['AvailableBalance'])

    if args.command == 'generate_hits':
        num_hits, examples_per_hit = get_num_hits(args)
        generate_hits(args, hits_dir, examples_per_hit)
    elif args.command == 'launch_hits':
        num_hits, examples_per_hit = get_num_hits(args)
        start = 0
        end = num_hits
        print(f'launching HITs {start} through {end - 1}')
        hit_xml_paths = []
        for hiti in range(start, end):
            fname = pth.join(hits_dir, f'hit_{hiti}.xml')
            if not pth.exists(fname):
                raise Exception(f'Need to run generate_hits for file {fname}')
            hit_xml_paths.append(fname)

        new_hit, hit_type = create_hit(args, mturk, hits_dir, hit_xml_paths, max_assignments)
        print('A new HIT has been created. You can preview it here:')
        print(f'{hit_base_url}' + new_hit['HIT']['HITGroupId'])
        print('HITID = ' + new_hit['HIT']['HITId'] + ' (Use to Get Results)')
        print('HITTypeId = ' + hit_type['HITTypeId'])
    elif args.command == 'status':
        show_status(args, mturk, hits_dir, hit_base_url)
    elif args.command == 'retrieve':
        retrieve_results(args, mturk, hits_dir)
    elif args.command == 'delete_hits':
        delete_hits(args, mturk, hits_dir)
    elif args.command == 'approve':
        approve_assignments(args, mturk, hits_dir)
    else:
        raise Exception(f'Uknown command {args.command}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

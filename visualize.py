import os
import os.path as pth

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from jinja2 import Template

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.questioner
import models.answerer

import misc.utilities as utils
import misc.dataloader
import misc.visualization as vis
from misc.dataloader import VQAPoolDataset
from misc.cub_dataloader import CUBPoolDataset
from misc.awa_dataloader import AWAPoolDataset

import options

from analyze import load_models




##################################
# Latent variable visualizations

def generate_z_info(batch, qbot, zsources, inferences, ind2word):
    qd = {}
    # NOTE: manual must be after None because it uses the results of None
    for zkind in zsources:
        for inference in inferences:
            if zkind[:6] == 'manual':
                if inference != ('sample', 'greedy'):
                    continue
                inference = ('manual', 'greedy')
                newz = vis.single_dim_manual_z(qd[('encoder',) + ('greedy', 'greedy')][1], qbot, dim=int(zkind[6:]))
                nsamples = len(newz)
                z_source = zkind
            elif zkind == 'interpolate':
                assert zsources[0] in ['policy', 'encoder'], 'determine where to start the interpolation from the policy or the encoder'
                assert ('greedy', 'greedy') in inferences, 'always start interpolation from greedy inference'
                # note that the z inference method is ignored but the
                # decoder inference method is listened to
                if inference[1] == 'sample': # generate multiple samples from the decoder
                    replications = 5
                elif inference[1] == 'greedy':
                    replications = 1
                z_source = 'manual'
                inference = ('interp', inference[1])
                newz = vis.interp_manual_z(qd[(zsources[0],) + ('greedy', 'greedy')][1], replications=replications)
                nsamples = len(newz)
            elif inference == ('greedy', 'greedy'):
                newz = None
                nsamples = 1
                z_source = zkind
            else:
                newz = None
                nsamples = 10
                z_source = zkind
            print(zkind, inference, qbot.varType)
            qd[(zkind,) + inference] = vis.get_questions(batch, qbot, z_source, inference, ind2word, nsamples=nsamples, manualz=newz)
    return qd

def render_column(qbot, questions, latents, k, exi, inference, collapse=False):
    ztype = k[1] if inference[0] == 'manual' else inference[0]
    dectype = inference[1]
    result = '<br><b>z-{}, dec-{}:</b><br>'.format(ztype, dectype)
    prev_question = ''
    prev_sample_i = 0
    for sample_i in range(len(questions)):
        new_question = questions[sample_i][exi]
        if collapse and new_question == prev_question:
            continue
        result += new_question + '<br>'
        if sample_i == 0:
            result += '(z=' + latents[sample_i][exi][0]
        else:
            curr_z = latents[sample_i][exi][1]
            prev_z = latents[prev_sample_i][exi][1]
            _, curr_idx = curr_z.reshape(qbot.num_vars, -1).max(dim=1)
            _, prev_idx = prev_z.reshape(qbot.num_vars, -1).max(dim=1)
            change_idxs = (curr_idx - prev_idx).nonzero().reshape(-1).tolist()
            result += '(zdiff=' + str(change_idxs)
            result += ')<br>'
            result += '(z=' + latents[sample_i][exi][0]
        result += ')<br>'
        prev_question = new_question
        prev_sample_i = sample_i
    return result                                              

def render_webpage(batch, qbot, qd, zsources, inferences, dataset):
    gt_ques_str = utils.old_idx_to_str(
                        dataset.ind2word,
                        batch['ques'],
                        batch['ques_len'],
                        batch['img_id_pool'], 0, [])
    gt_ques_str = [q[0].strip('<START> ').strip(' <END>')
                        for q in gt_ques_str]

    def render_partial_key(k, exi):
        ques = ''
        for z_inf, dec_inf in inferences:
            if k == 'interpolate':
                z_inf = 'interp'
            qdk = (k, z_inf, dec_inf)
            if qdk not in qd:
                continue
            questions = qd[qdk][0]
            latents = qd[qdk][1]
            ques += render_column(qbot, questions, latents, k, exi, (z_inf, dec_inf), collapse=(z_inf == 'interp'))
        return ques

    keys = ['policy', 'encoder', 'interpolate']
    #keys = ([('policy', 'encoder')] +
    #        [(zs,) for zs in zsources if zs is not None and 'manual' in zs] +
    #        [('manual',) for zs in zsources if zs == 'interpolate'] +
    #        [('prior',)])

    examples = []
    display_data = {
        'keys': keys,
        'examples': examples,
    }
    target_idxs = batch['target_pool'].view(-1).tolist()
    ans2label = dataset.ans2label
    ans2label['not relevant'] = qbot.ans_not_relevant_token
    label2ans = {label: ans for ans, label in ans2label.items()}

    for exi in range(len(batch['img_id_pool'])):
        img_paths = [vis.load_image(batch['img_id_pool'][exi][i].item())[1] for i in range(batch['img_id_pool'].shape[1])]
        try:
            # TODO: figure out why this fails sometimes
            answer = label2ans[int(batch['ansIdx'][exi, 0])]
        except:
            answer = '<FAILURE>'
        example = {
                'target': target_idxs[exi],            
                'answer': answer,
                'gt_ques': gt_ques_str[exi],
                'img_uris': [vis.img_to_datauri(img_path) for img_path in img_paths],
                'questions': [render_partial_key(k, exi) for k in keys],
            }
        
        examples.append(example)

    with open('templates/z_vis.html') as f:
        template = Template(f.read())
    html = template.render(display_data)
    return html


##################################
# Dialog rollout visualizations

def generate_and_render_dialog(batch, qbot, abot, dataset, numRounds, qexp, abf,
                               generate_only=False):
    with open('templates/dialog_viz.html') as f:
        dialog_template = Template(f.read())
    wrap_period = 4
    z_inference = 'sample'
    batch_size = batch['img_pool'].shape[0]
    pool_size = batch['img_id_pool'].shape[1]
    ans2label = dataset.ans2label
    ans2label['not relevant'] = qbot.ans_not_relevant_token
    label2ans = {label: ans for ans, label in ans2label.items()}
    assert len(ans2label) == len(label2ans)

    rounds = []

    qbot.reset()
    # observe the image
    qbot.observe(images=batch['img_pool'])
    qbot.tracking = True

    for Round in range(numRounds):
        print('Round {}'.format(Round))
        if Round == 0:
            # observe initial question
            qbot.observe(start_question=True)
            # since we only has 1 round.
            qbot.observe(start_answer=True)

        # decode the question
        ques, ques_len, _ = qbot.forwardDecode(dec_inference='sample',
                                               z_inference=z_inference,
                                               z_source='policy')

        region_attn = qbot.ctx_coder.region_attn.squeeze(3)
        region_uris = vis.region_att_images(batch['img_id_pool'],
                                            region_attn,
                                            batch['img_pool_spatial'],
                                            dataset.name)
        img_attn = qbot.ctx_coder.img_attn.squeeze(2)
        pool_uris = vis.pool_atten_uris(img_attn, wrap_period=wrap_period)

        logit = qbot.predictImage()
        _, predict = torch.max(logit, dim=1)
        predCorrect = (predict == batch['target_pool'].view(-1)).to(torch.float)

        predict_region_attn = qbot.predict_ctx.attn.squeeze(3)
        predict_region_uris = vis.region_att_images(batch['img_id_pool'],
                                                    predict_region_attn,
                                                    batch['img_pool_spatial'],
                                                    dataset.name)
        predict_probs = F.softmax(logit, dim=1)
        predict_uris = vis.pool_atten_uris(predict_probs,
                                           wrap_period=wrap_period)


        # observe the question here.
        qbot.observe(ques=ques, ques_len=ques_len, gt_ques=False)
        # answer the question here.
        ans, rel_logit = abot(batch['target_image'], ques, ques_len, inference_mode=False)
        _, ans_idx = torch.max(ans, dim=1)
        _, ans_rel_idx = torch.max(rel_logit, dim=1)
        # to predict the target image, use the latest latent state and the predict answer to select the target images.
        qbot.observe(ans=ans_idx, ans_rel=ans_rel_idx)
        ans_idx = qbot.answers[-1]

        rel_probs = F.softmax(rel_logit, dim=1)[:, 1].tolist()
        rel_uris = vis.pool_atten_uris(F.softmax(rel_logit, dim=1)[:, 1:2], wrap_period=wrap_period)
        
        gen_ques_str = utils.old_idx_to_str(dataset.ind2word, ques, ques_len, batch['img_id_pool'], 0, [])
        gen_ques_str = [q[0].strip('<START> ').strip(' <END>') for q in gen_ques_str]
        
        rounds.append({
            'questions': gen_ques_str,
            'answers': [label2ans[i] for i in ans_idx.tolist()],
            'preds': predict.tolist(),
            'region_uris': region_uris,
            'pool_atten_uris': pool_uris,
            'predict_region_uris': predict_region_uris,
            'predict_uris': predict_uris,
            'is_rel_probs': rel_probs,
            'rel_uris': rel_uris,
        })


    # render rollouts into a webpage
    examples = []
    target_idxs = batch['target_pool'].view(-1).tolist()
    # when displaying a pool, put this many images on one row then wrap to the next row

    for exi in range(batch_size):
        img_paths = []
        img_urls = []
        for i in range(pool_size):
            img_idx = batch['img_id_pool'][exi][i].item()
            assert img_idx != 0
            img_path = vis.load_image(img_idx, dataset_name=dataset.name)[1]
            img_url = vis.load_image_path(img_idx, dataset_name=dataset.name,
                                          local=False)
            img_paths.append(img_path)
            img_urls.append(img_url)
        example = {
            'target': target_idxs[exi],
            'img_uris': [vis.img_to_datauri(img_path) for img_path in img_paths],
            'img_urls': img_urls,
        }

        examples.append(example)

    if generate_only:
        return {
            'title': 'qbot {}, abot {}'.format(qexp, abf),
            'examples': examples,
            'rounds': rounds,
            'wrap_period': wrap_period,
        }

    html = dialog_template.render({
        'title': 'qbot {}, abot {}'.format(qexp, abf),
        'examples': examples,
        'rounds': rounds,
        'wrap_period': wrap_period,
    })
    return html



def main():
    params = options.readCommandLine()

    data_params = options.data_params(params)
    if params['dataset'] == 'VQA':
        dataset = VQAPoolDataset(data_params, ['train', 'val1', 'val2'])
        dataset.split = 'val1'
    elif params['dataset'] == 'CUB':
        dataset = CUBPoolDataset(data_params, ['train', 'val', 'test'],
                                 load_vis_info=True)
        dataset.split = 'val'
    elif params['dataset'] == 'AWA':
        dataset = AWAPoolDataset(data_params, ['train', 'val', 'test'],
                                 load_vis_info=True)
        dataset.split = 'val'

    dataloader = DataLoader(
                    dataset,
                    batch_size=params['batchSize'],
                    shuffle=False,
                    num_workers=10,
                    drop_last=True,
                    pin_memory=True)
    dliter = iter(dataloader)
    for _ in range(params['batchIndex'] + 1):
        batch = next(dliter)
    batch = {key: v.cuda() if hasattr(v, 'cuda') else v
                        for key, v in batch.items()}

    qbot, abot, q_exp, qbf, abf, qep, aep = next(iter(load_models(params)))

    if params['visMode'] == 'latent':
        zsources = ['encoder', 'policy']
        #zsources += ['interpolate']
        #zsources += ['manual{}'.format(dim) for dim in list(range(128))[:3]]
        #zsources += ['prior']
        inferences = [('greedy', 'greedy'),
                      ('greedy', 'sample'),
                      ('sample', 'greedy'),
                      ('sample', 'sample')]
        qd = generate_z_info(batch, qbot, zsources, inferences, dataset.ind2word)
        html = render_webpage(batch, qbot, qd, zsources, inferences, dataset)
        save_file = pth.join(params['savePath'],
                            '{}_z_examples.html'.format(params['evalSaveName']))
        with open(save_file, 'w') as f:
            f.write(html)

    elif params['visMode'] == 'interpolate':
        zsources = ['policy', 'interpolate']
        inferences = [('greedy', 'greedy'),
                      ('sample', 'greedy'),
                      ('sample', 'sample')]
        qd = generate_z_info(batch, qbot, zsources, inferences, dataset.ind2word)
        html = render_webpage(batch, qbot, qd, zsources, inferences, dataset)
        save_file = pth.join(params['savePath'],
                            '{}_z_examples.html'.format(params['evalSaveName']))
        with open(save_file, 'w') as f:
            f.write(html)

    elif params['visMode'] == 'dialog':
        html = generate_and_render_dialog(batch, qbot, abot, dataset, params['maxRounds'], q_exp, abf)
        save_file = pth.join(params['savePath'],
                            '{}_dialog_examples.html'.format(params['evalSaveName']))
        with open(save_file, 'w') as f:
            f.write(html)

    elif params['visMode'] == 'mturk':
        dialog_data = generate_and_render_dialog(batch, qbot, abot, dataset, params['maxRounds'], q_exp, abf, generate_only=True)
        os.makedirs(pth.join(params['savePath'], 'mturk'), exist_ok=True)
        save_file = pth.join(params['savePath'], 'mturk',
                            '{}_dialog_data.joblib'.format(params['evalSaveName']))
        joblib.dump(dialog_data, save_file, compress=True)


if __name__ == '__main__':
    main()

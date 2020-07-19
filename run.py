#!/usr/bin/env python3
import subprocess
from subprocess import Popen
import time
import tempfile
import socket
import argparse
import sys
import os
import os.path as pth
import glob
import shlex
import random

parser = argparse.ArgumentParser(description='Generic slurm run script')
parser.add_argument('EXP', type=str, nargs='?', help='experiment to run',
                    default=None)
parser.add_argument('EVAL', type=str, nargs='?', help='evaluation config string',
                    default=None)
parser.add_argument('-w', '--node', help='name of the machine to run this on')
parser.add_argument('-x', '--exclude', help='do not run on this machine')
parser.add_argument('-g', '--ngpus', type=int, default=1,
                    help='number of gpus to use')
parser.add_argument('-d', '--delay', type=int, default=0,
                    help='number of hours to delay job start')
parser.add_argument('-p', default='debug_noslurm',
                choices=['debug', 'short', 'long', 'batch', 'debug_noslurm'])
parser.add_argument('-f', '--profile', type=int, default=0)
parser.add_argument('-q', '--qos', help='slurm quality of service',
                        choices=['overcap'])
parser.add_argument('-m', '--mode', default='train',
                        help='which command to run',
                        choices=['train', 'analyze', 'visualize', 'train_lm',
                                 'mturk_generate_hits', 'mturk_launch_hits',
                                 'mturk_retrieve_hits', 'mturk_status',
                                 'mturk_delete_hits', 'mturk_approve',
                                 'check'])

args = parser.parse_args()

cmd_dir = '.data/cmds'
log_dir = 'logs/run_logs'
all_exp_dir = 'data/experiments'
Popen('mkdir -p {}'.format(cmd_dir), shell=True).wait()
Popen('mkdir -p {}'.format(log_dir), shell=True).wait()
Popen('mkdir -p {}'.format(all_exp_dir), shell=True).wait()
python = 'python -m torch.utils.bottleneck ' if args.profile == 1 else 'python'

jobid = 0
def runcmd(cmd, external_log=None):
    '''
    Run cmd, a string containing a command, in a bash shell using gpus
    allocated by slurm. Frequently changed slurm settings are configured
    here. 
    '''
    global jobid
    log_fname = '{}/job_{}_{:0>3d}_{}.log'.format(log_dir,
                    int(time.time()), random.randint(0, 999), jobid)
    if external_log:
        if pth.lexists(external_log):
            os.unlink(external_log)
        # realpath correctly deals with symlinks
        link_name = pth.relpath(pth.realpath(log_fname),
                                pth.realpath(pth.dirname(external_log)))
        os.symlink(link_name, external_log)
    jobid += 1
    # write SLURM job id then run the command
    write_slurm_id = True
    if write_slurm_id:
        script_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                             dir=cmd_dir, prefix='.', suffix='.slurm.sh')
        script_file.write('echo "slurm job id: $SLURM_JOB_ID"\n')
        script_file.write('echo ' + cmd + '\n')
        script_file.write('echo "host: $HOSTNAME"\n')
        script_file.write('echo "cuda: $CUDA_VISIBLE_DEVICES"\n')
        #script_file.write('nvidia-smi -i $CUDA_VISIBLE_DEVICES\n')
        script_file.write(cmd)
        script_file.close()
        # use this to restrict runs to current host
        #hostname = socket.gethostname()
        #cmd = ' -w {} bash '.format(hostname) + script_file.name
        cmd = 'bash ' + script_file.name

    srun_prefix = 'srun --gres gpu:{} '.format(args.ngpus)
    if args.node:
        srun_prefix += '-w {} '.format(args.node)
    if args.exclude:
        srun_prefix += '-x {} '.format(args.exclude)
    if args.delay:
        srun_prefix += '--begin=now+{}hours '.format(args.delay)
    if args.qos:
        srun_prefix += '--qos {} '.format(args.qos)

    ############################################################################
    # uncomment the appropriate line to configure how the command is run
    #print(cmd)
    if args.p == 'debug':
        # debug to terminal
        Popen(srun_prefix + '-p debug --pty ' + cmd, shell=True).wait()
        # debug to log file
        #Popen(srun_prefix + '-p debug -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)

        #logfile = open(log_fname, 'w', buffering=1)
        #proc = Popen(cmd, shell=True, stdout=logfile, stderr=logfile, bufsize=0, universal_newlines=True)
        #proc.wait()
        #for line in proc.stdout:
        #    print(line, end='')
        #    logfile.write(line)
        #proc.wait()
        return None
    elif args.p == 'debug_noslurm':
        # debug on current machine without slurm (manually set CUDA_VISIBLE_DEVICES)
        Popen(cmd, shell=True).wait()
        return None
    elif args.p == 'short':
        # log file
        Popen(srun_prefix + '-p short -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
        return log_fname
    elif args.p == 'long':
        # log file
        Popen(srun_prefix + '-p long -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
        return log_fname
    elif args.p == 'batch':
        # This is not a partition but is for use with sbatch, which itself
        # specifies a partition. The script which calls `python run.py` should
        # be run with sbatch and have the appropriate partition specified.
        # Furthermore, it is impotant to Popen.wait() for sbatch calls to srun.
        # log file
        Popen(srun_prefix + '-o {} --open-mode=append '.format(log_fname) + cmd, shell=True).wait()
        return log_fname


#######################################
# Config

def config(exp):
    # generic settings
    if exp is None:
        import warnings
        warnings.warn('Missing experiment code')
        return locals()
    experiment = exp # put it in locals()
    assert exp.startswith('exp')
    exp_dir = pth.join(all_exp_dir, exp)
    model_dir = pth.join(exp_dir, 'models')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_fname = pth.join(exp_dir, 'log.txt')
    test_log_fname = pth.join(exp_dir, 'analyze_log.txt')
    vers = list(map(int, exp[3:].split('.')))
    vers += [0] * (100 - len(vers))

    # project defaults
    trainMode = ''
    rounds = ''
    abot = ''
    qbot = ''
    batch = ''
    numWorkers = ''
    decVar = ''
    query = ''
    conditionEncoder = ''
    pool = ''
    varType = ''
    learningRate = ''
    freezeMode = ''
    zsource = ''
    evalLimit = ''
    coeff = ''
    mixing = ''
    dataset = ''
    dataset_id = 'VQA'
    speaker = ''
    numEpochs = ''
    mturkExps = ''
    numExamples = ''
    mturkEnv = ''
    hitType = ''
    vis_code = ''

    # pre-train qbot
    if vers[0] in [1]:
        trainMode = '-trainMode pre-qbot'
        # vs continuous random variable
        if vers[1] == 1:
            varType = '-varType cont -latentSize 512'
        elif vers[1] == 2:
            varType = '-varType cont -latentSize 128'
        elif vers[1] == 3:
            varType = '-num_embeddings 4 -num_vars 32'
        elif vers[1] == 4:
            varType = '-num_embeddings 4 -num_vars 128 -latentSize 32'
        elif vers[1] == 5:
            varType = '-rnnHiddenSize 256 -imgEmbedSize 256'
        elif vers[1] == 6:
            varType = '-rnnHiddenSize 256'
        # don't let the encoder see the pool
        if vers[2] == 1:
            conditionEncoder = '-conditionEncoder none'
        # pre-train for question generation only
        if vers[3] == 1:
            coeff = '-predictLossCoeff 0'

    # pre-train abot
    elif vers[0] == 2:
        trainMode = '-trainMode pre-abot'
        if vers[1] == 1:
            learningRate = '-scheduler plateau -learningRate 1e-3 -lrDecayRate 0.1 -minLRate 1e-8'

    # fine-tune qbot
    elif vers[0] == 3:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4'
        abot = '-astartFrom data/experiments/exp2/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp1.0.0/qbot_ep_64.vd'
        if vers[5] == 1:
            qbot = '-qstartFrom data/experiments/exp1.1.0/qbot_ep_64.vd'
        elif vers[5] == 2:
            qbot = '-qstartFrom data/experiments/exp1.0.1/qbot_ep_64.vd'
        elif vers[5] == 3:
            # no predict_loss during pre-training
            qbot = '-qstartFrom data/experiments/exp1.0.0.1/qbot_ep_64.vd'
        elif vers[5] == 4:
            qbot = '-qstartFrom data/experiments/exp1.2.0.0/qbot_ep_64.vd'
        elif vers[5] == 5:
            qbot = '-qstartFrom data/experiments/exp1.3.0.0/qbot_ep_64.vd'
        elif vers[5] == 6:
            qbot = '-qstartFrom data/experiments/exp1.4.0.0/qbot_ep_64.vd'
        elif vers[5] == 7:
            qbot = '-qstartFrom data/experiments/exp1.5.0.0/qbot_ep_64.vd'

        # fine-tune the decoder
        if vers[1] == 1:
            decVar = '-decoderVar gumbel'

        # only fine-tune question generation head, not image prediction head
        if vers[2] == 1:
            freezeMode = '-freezeMode predict'
        elif vers[2] == 2:
            freezeMode = '-freezeMode none'

        # consider different sources of z
        if vers[3] == 1:
            zsource = '-zSourceFine prior'
        elif vers[3] == 2:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')

        # pool types
        if vers[4] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[4] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[4] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[4] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

    # fine-tune qbot
    elif vers[0] == 4:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp1.0.0.0/qbot_ep_20.vd'
        if vers[5] == 1:
            qbot = '-qstartFrom data/experiments/exp1.0.0.0/qbot_ep_20.vd'
        if vers[1] == 1:
            learningRate = '-learningRate 1e-5'
        elif vers[1] == 2:
            qbot = '-qstartFrom data/experiments/exp1.0.0/qbot_ep_20.vd'
        elif vers[1] == 3:
            learningRate = '-learningRate 1e-5'
            qbot = '-qstartFrom data/experiments/exp1.0.0/qbot_ep_20.vd'
        elif vers[1] == 4:
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        elif vers[1] == 5:
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.9995 -minLRate 1e-6'
        elif vers[1] == 6:
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.9998 -minLRate 1e-6'
        elif vers[1] == 7:
            learningRate = '-scheduler plateau -learningRate 1e-4 -lrDecayRate 0.1 -minLRate 1e-8'
        elif vers[1] == 8:
            learningRate = '-scheduler plateau -learningRate 1e-4 -lrDecayRate 0.33 -minLRate 1e-8 -lrPatience 1 -lrCooldown 1'
        elif vers[1] == 9:
            learningRate = '-scheduler plateau -learningRate 1e-4 -lrDecayRate 0.33 -minLRate 1e-8 -lrPatience 1 -lrCooldown 1'
            mixing = '-mixing 5'
        elif vers[1] == 10:
            learningRate = '-scheduler plateau -learningRate 1e-4 -lrDecayRate 0.33 -minLRate 1e-8 -lrPatience 1 -lrCooldown 1'
            mixing = '-mixing 3'
        elif vers[1] == 11:
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
            coeff = '-qcycleLossCoeff 1.0'

        # only fine-tune question generation head, not image prediction head
        if vers[2] == 1:
            freezeMode = '-freezeMode predict'
        elif vers[2] == 2:
            freezeMode = '-freezeMode none'
        elif vers[2] == 3:
            freezeMode = '-freezeMode policy'

        # consider different sources of z
        if vers[3] == 1:
            zsource = '-zSourceFine prior'
        elif vers[3] == 2:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')

        # pool types
        if vers[4] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[4] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[4] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[4] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

    # fine-tune qbot from fixed speaker model
    elif vers[0] == 5:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'

        qbot = '-qstartFrom data/experiments/exp4.4.0.2.2.0/qbot_ep_15.vd'
        pool = '-poolSize 4 -poolType random'
        batch = '-batchSize 200'
        rounds = '-maxRounds 5'

        if vers[1] == 1:
            mixing = '-mixing 5'
        elif vers[1] == 2:
            mixing = '-mixing 10'

        if vers[2] == 1:
            freezeMode = '-freezeMode all_but_policy'
        elif vers[2] == 2:
            freezeMode = '-freezeMode all_but_ctx'
        elif vers[2] == 3:
            freezeMode = '-freezeMode policy'

        if vers[3] == 1:
            qbot = '-qstartFrom data/experiments/exp4.4.3.0.2.0/qbot_ep_15.vd'
        elif vers[3] == 2:
            qbot = '-qstartFrom data/experiments/exp4.4.3.0.0.0/qbot_ep_15.vd'
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[3] == 3:
            qbot = '-qstartFrom data/experiments/exp4.4.3.0.1.0/qbot_ep_15.vd'
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[3] == 4:
            qbot = '-qstartFrom data/experiments/exp4.4.3.0.3.0/qbot_ep_15.vd'
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

    elif vers[0] == 6:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'

        qbot = '-qstartFrom data/experiments/exp4.4.0.2.2.0/qbot_ep_15.vd'
        pool = '-poolSize 4 -poolType random'
        batch = '-batchSize 200'
        rounds = '-maxRounds 5'

        coeff = '-qcycleLossCoeff 1.0'
        if vers[5] == 1:
            coeff = '-qcycleLossCoeff 0.1'
        elif vers[5] == 2:
            coeff = '-qcycleLossCoeff 0.01'

        if vers[1] == 1:
            freezeMode = '-freezeMode all_but_vqg'
        elif vers[1] == 2:
            freezeMode = '-freezeMode policy'

        if vers[2] == 1:
            mixing = '-mixing 5'
        elif vers[2] == 2:
            mixing = '-mixing 10'

        if vers[3] == 1:
            coeff += ' -ewcLossCoeff 1.0'

        if vers[4] == 1:
            qbot = '-qstartFrom data/experiments/exp4.4.3.0.2.0/qbot_ep_15.vd'

    # pre-train qbot
    if vers[0] == 7:
        trainMode = '-trainMode pre-qbot'
        freezeMode = '-freezeMode none'

        if vers[1] == 1:
            query = '-queryType dialog_qa'
        elif vers[1] == 2:
            query = '-queryType dialog_qa'
            speaker = '-speakerType one_part'

        if vers[2] == 1:
            varType = '-varType cont -latentSize 512'
            learningRate = '-scheduler plateau -learningRate 0.001 -lrDecayRate 0.1 -minLRate 1e-8'

    # fine-tune qbot with fixed policy
    elif vers[0] == 8:
        trainMode = '-trainMode fine-qbot'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp7.1.0.0/qbot_ep_20.vd'
        if vers[4] == 1:
            qbot = '-qstartFrom data/experiments/exp7.1.1.0/qbot_ep_20.vd'
        elif vers[4] == 2:
            qbot = '-qstartFrom data/experiments/exp7.2.0.0/qbot_ep_20.vd'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        freezeMode = '-freezeMode policy'

        # pool types
        if vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        elif vers[1] == 11:
            pool = '-poolSize 2 -poolType easy'
            rounds = '-maxRounds 5'
        elif vers[1] == 12:
            pool = '-poolSize 4 -poolType easy'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 13:
            pool = '-poolSize 9 -poolType easy'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        elif vers[1] == 21:
            pool = '-poolSize 2 -poolType hard'
            rounds = '-maxRounds 5'
        elif vers[1] == 22:
            pool = '-poolSize 4 -poolType hard'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 23:
            pool = '-poolSize 9 -poolType hard'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        # fine-tune the decoder
        if vers[2] == 1:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode none'
        elif vers[2] == 2:
            coeff = '-qcycleLossCoeff 1.0'
            freezeMode = '-freezeMode all_but_vqg'
        elif vers[2] == 3:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')
        # pass gradients through the decoder, but don't fine-tune it
        elif vers[2] == 4:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode decoder'

        if vers[3] == 1:
            # TODO: use 2 instead of 1... this is just always policy???
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq 1'
        elif vers[3] == 2:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq 5'
        elif vers[3] == 3:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq 10'
        elif vers[3] == 4:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq 10'
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.9995 -minLRate 1e-6'
        elif vers[3] == 5:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq 2'
        elif vers[3] == 6:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq ep2'
        elif vers[3] == 7:
            freezeMode += ' -freezeMode2 all_but_policy -freezeMode2Freq ep5'

    # fine-tune qbot from fixed policy
    elif vers[0] == 9:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'

        # pool types
        if vers[1] == 0:
            qbot = '-qstartFrom data/experiments/exp8.0.0.0/qbot_ep_20.vd'
            if vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp8.0.0.0.2/qbot_ep_20.vd'
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            qbot = '-qstartFrom data/experiments/exp8.1.0.0/qbot_ep_20.vd'
            if vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp8.1.0.0.2/qbot_ep_20.vd'
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            qbot = '-qstartFrom data/experiments/exp8.2.0.0/qbot_ep_20.vd'
            if vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp8.2.0.0.2/qbot_ep_20.vd'
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            qbot = '-qstartFrom data/experiments/exp8.3.0.0/qbot_ep_20.vd'
            if vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp8.3.0.0.2/qbot_ep_20.vd'
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        elif vers[1] == 11:
            qbot = '-qstartFrom data/experiments/exp8.11.0.0/qbot_ep_10.vd'
            pool = '-poolSize 2 -poolType easy'
            rounds = '-maxRounds 5'
        elif vers[1] == 12:
            qbot = '-qstartFrom data/experiments/exp8.12.0.0/qbot_ep_10.vd'
            pool = '-poolSize 4 -poolType easy'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 13:
            qbot = '-qstartFrom data/experiments/exp8.13.0.0/qbot_ep_10.vd'
            pool = '-poolSize 9 -poolType easy'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        if vers[1] == 20:
            qbot = '-qstartFrom data/experiments/exp8.0.0.0.1/qbot_ep_20.vd'
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 21:
            qbot = '-qstartFrom data/experiments/exp8.1.0.0.1/qbot_ep_20.vd'
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 22:
            qbot = '-qstartFrom data/experiments/exp8.2.0.0.1/qbot_ep_20.vd'
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 23:
            qbot = '-qstartFrom data/experiments/exp8.3.0.0.1/qbot_ep_20.vd'
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        if vers[2] == 1:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode none'

    # fine-tune for CUB/AWA with fixed policy
    elif vers[0] == 10:
        trainMode = '-trainMode fine-qbot'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp7.1.0.0/qbot_ep_20.vd'
        if vers[4] == 2:
            qbot = '-qstartFrom data/experiments/exp7.2.0.0/qbot_ep_20.vd'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        freezeMode = '-freezeMode policy'
        dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
        if vers[3] == 1:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'

        if vers[2] == 3:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')
        # pass gradients through the decoder, but don't fine-tune it
        elif vers[2] == 4:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode decoder'

        # pool types
        if vers[1] in [0, 1]: # default (0) + consistent with other codes (1)
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

    # fine-tune policy for CUB/AWA
    elif vers[0] == 11:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
        if vers[4] == 2:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'

        # pool types
        if vers[1] in [0, 1]: # default (0) + consistent with other codes (1)
            qbot = '-qstartFrom data/experiments/exp10.1.0.0/qbot_ep_20.vd'
            if vers[3] == 1:
                qbot = '-qstartFrom data/experiments/exp10.1.0.1/qbot_ep_10.vd'
            elif vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp10.1.0.1.2/qbot_ep_10.vd'
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            qbot = '-qstartFrom data/experiments/exp10.2.0.0/qbot_ep_20.vd'
            if vers[3] == 1:
                qbot = '-qstartFrom data/experiments/exp10.2.0.1/qbot_ep_10.vd'
            elif vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp10.2.0.1.2/qbot_ep_10.vd'
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            qbot = '-qstartFrom data/experiments/exp10.3.0.0/qbot_ep_20.vd'
            if vers[3] == 1:
                qbot = '-qstartFrom data/experiments/exp10.3.0.1/qbot_ep_10.vd'
            elif vers[4] == 2:
                qbot = '-qstartFrom data/experiments/exp10.3.0.1.2/qbot_ep_10.vd'
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        if vers[2] == 1:
            learningRate = '-learningRate 1e-5'

    elif vers[0] in [12, 13, 14]:
        pass # used elsewhere

    # stage 1: pre-train qbot
    elif vers[0] == 15:
        trainMode = '-trainMode pre-qbot'
        freezeMode = '-freezeMode none'
        query = '-queryType dialog_qa'
        speaker = '-speakerType two_part_zgrad'
        if vers[2] == 1:
            speaker = '-speakerType two_part_zgrad_logit'
        elif vers[2] == 2:
            speaker = '-speakerType two_part_zgrad_codebook'

        if vers[1] == 1:
            varType = '-varType cont -latentSize 512'
            learningRate = '-scheduler plateau -learningRate 0.001 -lrDecayRate 0.1 -minLRate 1e-8'
        elif vers[1] == 2:
            varType = '-varType none -latentSize 512'
            learningRate = '-scheduler plateau -learningRate 0.001 -lrDecayRate 0.1 -minLRate 1e-8'
        elif vers[1] == 3:
            varType = '-varType none -latentSize 512'
            learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'

    # stage 2: fine-tune qbot with fixed policy
    elif vers[0] == 16:
        trainMode = '-trainMode fine-qbot'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp15.0.0.0/qbot_ep_20.vd'
        # vers[4] == 0 -> sample z feedback
        if vers[4] == 1:
            # logit feedback
            qbot = '-qstartFrom data/experiments/exp15.0.1.0/qbot_ep_20.vd'
        elif vers[4] == 2:
            # codebook feedback
            qbot = '-qstartFrom data/experiments/exp15.0.2.0/qbot_ep_20.vd'
        elif vers[4] == 3:
            # continuous baseline
            qbot = '-qstartFrom data/experiments/exp15.1.0.0/qbot_ep_20.vd'
        elif vers[4] == 4:
            # non-variational continuous baseline
            qbot = '-qstartFrom data/experiments/exp15.2.0.0/qbot_ep_20.vd'
        elif vers[4] == 5:
            # logit feedback from scratch
            qbot = '-qstartFrom data/experiments/exp15.0.1.0/qbot_ep_0.vd'
        elif vers[4] == 6:
            # continuous baseline from scratch
            qbot = '-qstartFrom data/experiments/exp15.1.0.0/qbot_ep_0.vd'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        if vers[5] == 1:
            learningRate = '-learningRate 1e-2 -lrDecayRate 0.99 -minLRate 1e-6'
        elif vers[5] == 2:
            learningRate = '-learningRate 1e-3 -lrDecayRate 0.999 -minLRate 1e-6'
        freezeMode = '-freezeMode policy'

        # pool types
        if vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        elif vers[1] == 21:
            pool = '-poolSize 2 -poolType hard'
            rounds = '-maxRounds 5'
        elif vers[1] == 22:
            pool = '-poolSize 4 -poolType hard'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 23:
            pool = '-poolSize 9 -poolType hard'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        # fine-tune the decoder
        if vers[2] == 1:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode none'
        # separate speaker baseline
        elif vers[2] == 2:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')

        if vers[3] == 1:
            dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
            dataset_id = 'CUB'
        elif vers[3] == 2:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
            dataset_id = 'AWA'

    # stage 3: fine-tune qbot from fixed policy
    elif vers[0] == 17:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'

        # pool types
        qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.0/qbot_ep_20.vd'
        if vers[2] == 1:
            # sample z feedback
            qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.{vers[3]}/qbot_ep_20.vd'
        elif vers[2] == 2:
            # logit feedback
            qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.{vers[3]}.1/qbot_ep_20.vd'
        elif vers[2] == 3:
            # codebook feedback
            qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.{vers[3]}.2/qbot_ep_20.vd'
        elif vers[2] == 4:
            # continuous baseline
            qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.{vers[3]}.3/qbot_ep_20.vd'
        elif vers[2] == 5:
            # non-variational continuous baseline
            qbot = f'-qstartFrom data/experiments/exp16.{vers[1]}.0.{vers[3]}.4/qbot_ep_20.vd'
        if vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        if vers[3] == 1:
            dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
            dataset_id = 'CUB'
        elif vers[3] == 2:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
            dataset_id = 'AWA'

        if vers[4] == 1:
            learningRate = '-scheduler plateau -learningRate 0.001 -lrDecayRate 0.1 -minLRate 1e-8'
        elif vers[4] == 2:
            learningRate = '-scheduler plateau -learningRate 3e-5 -lrDecayRate 0.9999 -minLRate 1e-8'
        elif vers[4] == 3:
            learningRate = '-learningRate 3e-5 -lrDecayRate 0.9995 -minLRate 1e-8'
        elif vers[4] == 4:
            learningRate = '-learningRate 1e-2 -lrDecayRate 0.99 -minLRate 1e-6'
        elif vers[4] == 5:
            learningRate = '-learningRate 1e-3 -lrDecayRate 0.999 -minLRate 1e-6'

    # train question language model
    elif vers[0] == 18:
        assert args.mode != 'train'
        learningRate = '-learningRate 1e-3'
        numEpochs = '-numEpochs 50'

    # stage 2 after length bug fix
    elif vers[0] == 19:
        trainMode = '-trainMode fine-qbot'
        abot = '-astartFrom data/experiments/exp2.0.0.0/abot_ep_15.vd'
        qbot = '-qstartFrom data/experiments/exp15.0.0.0.0/qbot_ep_20.vd'
        # vers[4] == 0 -> sample z feedback
        if vers[4] == 1:
            # logit feedback
            qbot = '-qstartFrom data/experiments/exp15.0.1.0.0/qbot_ep_20.vd'
        elif vers[4] == 2:
            # codebook feedback
            qbot = '-qstartFrom data/experiments/exp15.0.2.0.0/qbot_ep_20.vd'
        elif vers[4] == 3:
            # continuous baseline
            qbot = '-qstartFrom data/experiments/exp15.1.0.0.0/qbot_ep_20.vd'
        elif vers[4] == 4:
            # non-variational continuous baseline
            qbot = '-qstartFrom data/experiments/exp15.2.0.0.0/qbot_ep_20.vd'
        elif vers[4] == 5:
            # logit feedback from scratch
            qbot = '-qstartFrom data/experiments/exp15.0.1.0.0/qbot_ep_0.vd'
        elif vers[4] == 6:
            # continuous baseline from scratch
            qbot = '-qstartFrom data/experiments/exp15.1.0.0.0/qbot_ep_0.vd'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        if vers[5] == 1:
            learningRate = '-learningRate 1e-2 -lrDecayRate 0.99 -minLRate 1e-6'
        elif vers[5] == 2:
            learningRate = '-learningRate 1e-3 -lrDecayRate 0.999 -minLRate 1e-6'
        freezeMode = '-freezeMode policy'

        # pool types
        if vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'
        elif vers[1] == 21:
            pool = '-poolSize 2 -poolType hard'
            rounds = '-maxRounds 5'
        elif vers[1] == 22:
            pool = '-poolSize 4 -poolType hard'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 23:
            pool = '-poolSize 9 -poolType hard'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        # fine-tune the decoder
        if vers[2] == 1:
            decVar = '-decoderVar gumbel'
            freezeMode = '-freezeMode none'
        # separate speaker baseline
        elif vers[2] == 2:
            zsource = '-zSourceFine speaker'
            zsource += ' ' + qbot.replace('-qstartFrom', '-speakerStartFrom')
        # directly go to stage 2.b
        elif vers[2] == 3:
            freezeMode = '-freezeMode decoder'

        if vers[3] == 1:
            dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
            dataset_id = 'CUB'
        elif vers[3] == 2:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
            dataset_id = 'AWA'

    # stage 3 after length bug fix
    elif vers[0] == 20:
        trainMode = '-trainMode fine-qbot'
        learningRate = '-learningRate 1e-4 -lrDecayRate 0.999 -minLRate 1e-6'
        abot = '-astartFrom data/experiments/exp2.0.0.0/abot_ep_15.vd'

        # pool types
        qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.0/qbot_ep_20.vd'
        if vers[2] == 1:
            # sample z feedback
            qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.{vers[3]}/qbot_ep_20.vd'
        elif vers[2] == 2:
            # logit feedback
            qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.{vers[3]}.1/qbot_ep_20.vd'
        elif vers[2] == 3:
            # codebook feedback
            qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.{vers[3]}.2/qbot_ep_20.vd'
        elif vers[2] == 4:
            # continuous baseline
            qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.{vers[3]}.3/qbot_ep_20.vd'
        elif vers[2] == 5:
            # non-variational continuous baseline
            qbot = f'-qstartFrom data/experiments/exp19.{vers[1]}.0.{vers[3]}.4/qbot_ep_20.vd'
        if vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            batch = '-batchSize 200'
            rounds = '-maxRounds 5'
        elif vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            batch = '-batchSize 64'
            rounds = '-maxRounds 9'

        if vers[3] == 1:
            dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
            dataset_id = 'CUB'
        elif vers[3] == 2:
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
            dataset_id = 'AWA'

        if vers[4] == 1:
            learningRate = '-scheduler plateau -learningRate 0.001 -lrDecayRate 0.1 -minLRate 1e-8'
        elif vers[4] == 2:
            learningRate = '-scheduler plateau -learningRate 3e-5 -lrDecayRate 0.9999 -minLRate 1e-8'
        elif vers[4] == 3:
            learningRate = '-learningRate 3e-5 -lrDecayRate 0.9995 -minLRate 1e-8'
        elif vers[4] == 4:
            learningRate = '-learningRate 1e-2 -lrDecayRate 0.99 -minLRate 1e-6'
        elif vers[4] == 5:
            learningRate = '-learningRate 1e-3 -lrDecayRate 0.999 -minLRate 1e-6'

    # mturk experiments
    elif vers[0] == 21:
        assert args.mode != 'train'
        mturkExps = '-expCodes exp15.0.1.0.0,exp19.2.1.0.4,exp20.2.2.0'
        vis_code = '-visCode eval3.{batch_index}.2.1.3'

        if vers[1] == 1:
            numExamples = '-numExamples 300'
            mturkEnv = '-mturkEnv production'
        elif vers[1] == 2:
            numExamples = '-numExamples 300'

        if vers[2] == 1:
            # language
            hitType = '-hitType question_comparison_v1'
            vis_code = '-visCode eval3.{batch_index}.2.1.3.1'
        elif vers[2] == 2:
            # grounding
            hitType = '-hitType question_comparison_v2'
            vis_code = '-visCode eval3.{batch_index}.2.1.3.1'


    if -1 in vers:
        batch = '-batchSize 10'
        numWorkers = '-numWorkers 0'
        evalLimit = '-evalLimit 20'
        numExamples = '-numExamples 10'

    if args.profile:
        pass

    return locals()


def eval_config(eval_code, ):
    # evaluation configuration
    # This should have access to everything set in config(exp) and should be
    # careful about overwriting any of it.
    if eval_code is None:
        if args.mode not in ['train', 'train_lm', 'check',
                             'mturk_generate_hits', 'mturk_launch_hits',
                             'mturk_retrieve_hits', 'mturk_status',
                             'mturk_delete_hits', 'mturk_approve']:
            raise Exception('Missing evaluation code')
        else:
            return locals()
    assert eval_code.startswith('eval')
    # TODO: this is redundant (just evaluate if an eval_code is present)
    assert args.mode in ['analyze', 'visualize'], 'In train mode, but eval code provided'
    eval_log_fname = pth.join(exp_dir, '{}_log.txt'.format(eval_code))
    eval_vers = list(map(int, eval_code[4:].split('.')))
    eval_vers += [0] * (100 - len(eval_vers))

    # project defaults (overrides vars from config())
    visMode = ''
    zsource = train_config['zsource']
    source_epoch = ''
    lm = ''
    batch_index = ''
    eval_split = ''
    eval_inf = ''

    # compute final metrics on val
    if eval_vers[0] == 1:
        assert args.mode == 'analyze'
        # provide a default abot if not set by vers[:]
        if vers[0] in [7, 15]:
            abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        batch = '-batchSize 256'
        rounds = '-maxRounds 5'
        lm = '-languageModel data/experiments/exp18.0.0.-2/lm_ep_3.vd'

        if eval_vers[1] == 0:
            pool = '-poolSize 2 -poolType contrast'
        elif eval_vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
        elif eval_vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
        elif eval_vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            rounds = '-maxRounds 9'
        elif eval_vers[1] == 11:
            pool = '-poolSize 2 -poolType easy'
        elif eval_vers[1] == 12:
            pool = '-poolSize 4 -poolType easy'
        elif eval_vers[1] == 13:
            pool = '-poolSize 9 -poolType easy'
            rounds = '-maxRounds 9'
        elif eval_vers[1] == 21:
            pool = '-poolSize 2 -poolType hard'
        elif eval_vers[1] == 22:
            pool = '-poolSize 4 -poolType hard'
        elif eval_vers[1] == 23:
            pool = '-poolSize 9 -poolType hard'
            rounds = '-maxRounds 9'

        # zsource = '-zSourceFine policy' Do not manually specify this because it would override the previous setting.
        # useful for getting the number of question matches
        if eval_vers[2] == 1:
            zsource = '-zSourceFine encoder'
            rounds = '-maxRounds 1'
        elif eval_vers[2] == 2:
            zsource = '-zSourceFine policy'
            rounds = '-maxRounds 1'

    # generate visualizations
    elif eval_vers[0] == 2:
        assert args.mode == 'visualize'
        batch = '-batchSize 300'
        if eval_vers[1] == 0:
            visMode = '-visMode latent'
        elif eval_vers[1] == 1:
            visMode = '-visMode dialog'
            if vers[0] == 15:
                rounds = '-maxRounds 5'
        if vers[0] in [7, 15]:
            abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        # default to whatever the exp was trained with
        if eval_vers[2] == 4:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 2:
            pool = '-poolSize 4 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 3:
            pool = '-poolSize 9 -poolType random'
            rounds = '-maxRounds 9'

    # generate mturk data
    elif eval_vers[0] == 3:
        batch = '-batchSize 100'
        visMode = '-visMode mturk'
        if vers[0] == 15:
            rounds = '-maxRounds 5'
        if vers[0] in [7, 15]:
            abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        batch_index = f'-batchIndex {eval_vers[1]}'
        # default to whatever the exp was trained with
        if eval_vers[2] == 4:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 2:
            pool = '-poolSize 4 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[2] == 3:
            pool = '-poolSize 9 -poolType random'
            rounds = '-maxRounds 9'
        if eval_vers[5] == 1:
            if dataset_id == 'VQA':
                eval_split = '-evalSplit val2'
            elif dataset_id in ['CUB', 'AWA']:
                eval_split = '-evalSplit test'
            else:
                raise Exception('Unknown dataset')
            eval_inf = '-evalInf sample,argmax'

    # compute final metrics on test
    elif eval_vers[0] == 4:
        assert args.mode == 'analyze'
        # provide a default abot if not set by vers[:]
        if vers[0] in [15]:
            abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        batch = '-batchSize 256'
        rounds = '-maxRounds 5'
        if dataset_id == 'VQA':
            eval_split = '-evalSplit val2'
        elif dataset_id in ['CUB', 'AWA']:
            eval_split = '-evalSplit test'
        else:
            raise Exception('Unknown dataset')
        lm = '-languageModel data/experiments/exp18.0.0.-2/lm_ep_3.vd'

        # by default use the model's parameters
        if eval_vers[1] == 4:
            pool = '-poolSize 2 -poolType contrast'
        elif eval_vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
        elif eval_vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
        elif eval_vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            rounds = '-maxRounds 9'

        if eval_vers[2] == 1:
            eval_inf = '-evalInf sample,argmax'
        elif eval_vers[2] == 2:
            eval_inf = '-evalInf argmax,sample'
        elif eval_vers[2] == 3:
            eval_inf = '-evalInf argmax,argmax'

        # zsource = '-zSourceFine policy' Do not manually specify this because it would override the previous setting.
        # useful for getting the number of question matches
        if eval_vers[2] == 1:
            zsource = '-zSourceFine encoder'
            rounds = '-maxRounds 1'
        elif eval_vers[2] == 2:
            zsource = '-zSourceFine policy'
            rounds = '-maxRounds 1'

    # generate visualizations
    elif eval_vers[0] == 5:
        assert args.mode == 'visualize'
        batch = '-batchSize 200'
        visMode = '-visMode dialog'
        if eval_vers[5] == 1:
            visMode = '-visMode latent'
        elif eval_vers[5] == 2:
            visMode = '-visMode interpolate'
        if vers[0] == 15:
            rounds = '-maxRounds 5'
            abot = '-astartFrom data/experiments/exp2.0.0/abot_ep_15.vd'
        # default to whatever the exp was trained with
        if eval_vers[1] == 4:
            pool = '-poolSize 2 -poolType contrast'
            rounds = '-maxRounds 5'
        elif eval_vers[1] == 1:
            pool = '-poolSize 2 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[1] == 2:
            pool = '-poolSize 4 -poolType random'
            rounds = '-maxRounds 5'
        elif eval_vers[1] == 3:
            pool = '-poolSize 9 -poolType random'
            rounds = '-maxRounds 9'

        if eval_vers[2] == 1:
            eval_inf = '-evalInf sample,argmax'
        elif eval_vers[2] == 2:
            eval_inf = '-evalInf argmax,sample'
        elif eval_vers[2] == 3:
            eval_inf = '-evalInf argmax,argmax'


    if eval_vers[4] == 1: # force CUB evaluation
        dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
    elif eval_vers[4] == 2: # force AWA
        dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
    elif eval_vers[4] == 3:
        if dataset_id == 'VQA':
            dataset = ''
        elif dataset_id == 'CUB':
            dataset = '-dataset CUB -poolInfo data/cub_pool_info_v1.pkl -inputImg data/cub_train36.hdf5'
        elif dataset_id == 'AWA':
            dataset = '-dataset AWA -poolInfo data/awa_pool_info_v1.pkl -inputImg data/AWA_train36.hdf5'
        else:
            raise Exception('Non-explicit dataset specification')
    # (note position 3 specifies epoch and is shared... :/)
    # by default, use the highest epoch checkpoint at the time of evaluation
    if eval_vers[3] == 1:
        if vers[0] in [15, 16, 19]:
            epoch = 20
        elif vers[0] in [17, 20]:
            epoch = 5
        else:
            raise Exception(f'Uknown epoch for experiment {vers}')
        source_epoch = f'-numEpochs {epoch}'
    elif eval_vers[3] == 2:
        source_epoch = f'-numEpochs 45'
    elif eval_vers[3] == 3:
        source_epoch = f'-numEpochs 64'
    elif eval_vers[3] == 4:
        source_epoch = f'-numEpochs 0'
    elif eval_vers[3] == 5:
        source_epoch = f'-numEpochs 5'


    if -1 in eval_vers:
        batch = '-batchSize 10'
        numWorkers = '-numWorkers 0'
        evalLimit = '-evalLimit 100'

    return locals()


train_config = config(args.EXP)
locals().update(train_config)

locals().update(eval_config(args.EVAL))


#######################################
# Actually run things

if args.mode == 'train':
    runcmd(('{python} train.py -useRedis 1 -useGPU \
            -savePath {all_exp_dir} \
            -saveName {experiment} \
            {dataset} \
            {batch} {numWorkers} {learningRate} {numEpochs} \
            {coeff} \
            {trainMode} {mixing} {rounds} {pool} \
            {abot} {qbot} \
            {varType} {decVar} {query} {speaker} \
            {conditionEncoder} \
            {freezeMode} {zsource} \
            {evalLimit}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'analyze':
    runcmd(('{python} analyze.py -useRedis 1 -useGPU \
            -savePath {all_exp_dir} \
            -saveName {experiment} \
            -evalSaveName {eval_code} \
            {eval_split} {eval_inf} \
            {lm} \
            {dataset} \
            {source_epoch} \
            {batch} {numWorkers} \
            {trainMode} {rounds} {pool} \
            {abot} \
            {zsource} \
            {evalLimit}').format(**locals()),
            external_log=eval_log_fname)
elif args.mode == 'visualize':
    runcmd(('{python} visualize.py -useRedis 1 -useGPU \
            -savePath {all_exp_dir} \
            -saveName {experiment} \
            -evalSaveName {eval_code} \
            {eval_split} {eval_inf} \
            {dataset} \
            {source_epoch} \
            {batch} {batch_index} \
            {rounds} {pool} \
            {abot} \
            {zsource} \
            {visMode}').format(**locals()),
            external_log=eval_log_fname)
elif args.mode == 'prepro':
    runcmd(f'{python} tools/generate_splits.py')
    runcmd(f'{python} tools/generate_pool.py')
elif args.mode == 'train_lm':
    runcmd(('{python} train_lm.py -useGPU \
            -savePath {all_exp_dir} \
            -saveName {experiment} \
            {batch} {numWorkers} {learningRate} {numEpochs}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_generate_hits':
    runcmd(('{python} mturk.py generate_hits {exp_dir} {mturkEnv} '
            '{vis_code} '
            '{numExamples} {mturkExps} {hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_launch_hits':
    runcmd(('{python} mturk.py launch_hits {exp_dir} {mturkEnv} '
            '{vis_code} '
            '{numExamples} {mturkExps} {hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_retrieve_hits':
    runcmd(('{python} mturk.py retrieve {exp_dir} {mturkEnv} '
            '{hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_approve':
    runcmd(('{python} mturk.py approve {exp_dir} {mturkEnv} '
            '{hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_delete_hits':
    runcmd(('{python} mturk.py delete_hits {exp_dir} {mturkEnv} '
            '{hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'mturk_status':
    runcmd(('{python} mturk.py status {exp_dir} {mturkEnv} '
            '{hitType}').format(**locals()),
            external_log=log_fname)
elif args.mode == 'check':
    if vers[0] in [15, 16, 19]:
        ep = 20
    elif vers[0] in [17, 20]:
        ep = 5
    print(exp, pth.exists(pth.join(exp_dir, f'qbot_ep_{ep}.vd')))
    #print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH', exp)
    #Popen(f'tail -n 20 {pth.join(exp_dir, "log.txt")}', shell=True).wait()

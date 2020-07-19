import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime



def readCommandLine(argv=None):
    parser = argparse.ArgumentParser(description='Train and Test the Visual Dialog model',
                                     formatter_class=argparse.RawTextHelpFormatter)

    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-dataset', default='VQA', choices=['VQA', 'CUB', 'AWA'])
    # VQA
    parser.add_argument('-inputImg', default='data/img_bottom_up_gzip1_chunked1.h5',
                            help='HDF5 file with image features')
    parser.add_argument('-inputQues', default='data/v2_vqa_data.h5',
                            help='HDF5 file with preprocessed questions')
    parser.add_argument('-inputJson', default='data/v2_vqa_info.json',
                            help='JSON file with info and vocab')
    parser.add_argument('-cocoDir', default='',
                            help='Directory for coco images, optional')
    parser.add_argument('-cocoInfo', default='',
                            help='JSON file with coco split information')
    parser.add_argument('-splitInfo', default='data/split_info_v3.json')
    parser.add_argument('-poolInfo', default='data/pool_info_v4.pkl')

    # CUB
    parser.add_argument('-cubInfo', default='data/cub_meta.json')

    # AWA
    parser.add_argument('-awaInfo', default='data/awa_meta.json')

    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-verbose', type=int, default=1,
                            help='Level of verbosity (default 1 prints some info)',
                            choices=[1, 2])
    parser.add_argument('-savePath', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-evalSaveName')
    parser.add_argument('-saveName', default='',
                            help='Name of save directory within savePath')
    parser.add_argument('-enable_log', action='store_true',
                            help='whether to save log files')
    parser.add_argument('-qstartFrom', type=str, default='',
                            help='Copy weights from model at this path')
    parser.add_argument('-speakerStartFrom', type=str,
                            help='Load speaker model from here')
    parser.add_argument('-astartFrom', type=str, default='',
                            help='Copy weights from model at this path')
    parser.add_argument('-continue', action='store_true',
                            help='Continue training from last epoch')

    #-------------------------------------------------------------------------
    # Model params
    parser.add_argument('-maxRounds', default=10, type=int,
                            help='default maxRound of the dialog')

    parser.add_argument('-randomSeed', default=32, type=int,
                            help='Seed for random number generators')

    parser.add_argument('-imgEmbedSize', default=512, type=int,
                            help='Size of the multimodal embedding')
    parser.add_argument('-imgFeatureSize', default=2048, type=int,
                            help='Size of the image feature')
    parser.add_argument('-embedSize', default=512, type=int,
                            help='Size of input word embeddings')

    parser.add_argument('-rnnHiddenSize', default=512, type=int,
                            help='Size of the LSTM state')
    parser.add_argument('-numLayers', default=1, type=int,
                            help='Number of layers in LSTM')
    
    parser.add_argument('-seqLength', default=20, type=int,
                            help='seq Length for the question')
    parser.add_argument('-ansSeqLength', default=3, type=int,
                            help='seq Length for the question')

    parser.add_argument('-ansObjective', default='both', type=str,
                            help='whether predict the question relavency with random question',
                            choices=['single', 'both'])    

    parser.add_argument('-ansEmbedSize', default=300, type=int,
                            help='latent answer embedding size for predictor')
    parser.add_argument('-wordDropoutRate', default=0.3, type=float,
                            help='word_dropout_rate for the decoder')

    # latent variable config
    parser.add_argument('-varType', default='gumbel', type=str,
                            help='Which type of latent variable:\n'
                                'cont - continuous with standard normal prior\n'
                                'gumbel - discrete via straight through gumbel softmax\n'
                                'none - do not put a variational prior on z',
                            choices=['cont', 'gumbel', 'none'])
    parser.add_argument('-latentSize', default=64, type=int,
                            help='latentSize for the VAE model')
    # discrete gumbel var
    parser.add_argument('-num_embeddings', type=int, default=4,
                        help='K: z uses K-way Categorical variables')
    parser.add_argument('-num_vars', type=int, default=128,
                        help='N: z has N K-way Categorical variables')
    # for gumbel-vae, defaults follow https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    parser.add_argument('-temp', type=float, default=1.0,
                        help='initial gumbel vae temp')
    parser.add_argument('-tempAnnealRate', type=float, default=0.00003)
    parser.add_argument('-tempMin', type=float, default=0.5)
    parser.add_argument('-softStartIters', type=int, default=10000,
                        help='for gumbelst, use soft for this many iterations')
    # decoder var type
    parser.add_argument('-decoderVar', default='cat', type=str,
                            help='What kind of inference should the decoder use?\n'
                                 'cat - categorical dist over words (no gradients)\n'
                                 'gumbel - gumbel softmax over words (allows gradients)',
                            choices=['cat', 'gumbel'])

    # KL loss term parameters
    parser.add_argument('-annealFunction', default='logistic',
                            help='anneal the KL weight (only applies to '
                                 'vaeMode=vae)',
                            choices=['logistic', 'linear'])
    parser.add_argument('-klWeight', default=0.1, type=float,
                            help='initial kl weight')
    parser.add_argument('-klWeightCC', default=0.0, type=float,
                            help='initial kl weight for context coder')
    parser.add_argument('-k', type=float, default=0.0025)
    parser.add_argument('-x0', type=int, default=2500)

    parser.add_argument('-conditionEncoder', type=str, default='images',
                        help='Whether allow questions based on the image pool',
                        choices=['images', 'none'])

    parser.add_argument('-queryType', type=str, default='dialog_only',
                        help='What information queries image features for question asking?\n'
                             'dialog_only - Just use the query output of dialogRNN\n'
                             'dialog_qa - Use the query from dialogRNN with the previous QA pair',
                        choices=['dialog_only', 'dialog_qa'])
    parser.add_argument('-speakerType', type=str, default='two_part',
                        choices=['two_part', 'one_part', 'two_part_zgrad',
                                 'two_part_zgrad_logit',
                                 'two_part_zgrad_codebook'])

    #-------------------------------------------------------------------------
    # Optimization / training params
    parser.add_argument('-poolType', default='contrast', type=str,
                            help='how to construct the pool',
                             choices=['contrast', 'random', 'hard', 'easy', 'contrast-random'])

    parser.add_argument('-poolSize', default=2, type=int,
                            help='pool size for question generation')

    parser.add_argument('-trainMode', default='sl-qbot',
                            help='What should train.py do? \n'
                            'pre-qbot - pre-train qbot using VQA2 synthetic pools\n'
                            'pre-abot - pre-train a standard abot for VQA\n'
                            'fine-qbot - fine-tune for guesswhich performance',
                            choices=['pre-qbot', 'pre-abot', 'fine-qbot'])
    parser.add_argument('-mixing', default=0, type=int,
                            help='Mix pre-training and fine-tuning batches. '
                            'If set to N!=0 then take a supervised pre-training '
                            'step every N iterations')

    freeze_mode_choices = ['decoder', 'predict', 'none', 'all',
                           'all_but_policy', 'all_but_ctx', 'policy',
                           'all_but_vqg']
    parser.add_argument('-freezeMode', default='decoder', type=str,
                            help='Which variables to fix during fine-tuning.\n'
                            'decoder - just fix the decoder\n'
                            'none - do not even fix the decoder (but do fix the speaker, if present)\n'
                            'all - do not learn anything\n'
                            'all_but_policy - only learn to ask different questions\n'
                            'all_but_ctx - only learn the context coder\n'
                            'all_but_vqg - only learn the question generator\n'
                            'policy - freeze the policy and the decoder\n'
                            'predict - fix the decoder and the pool predictor head',
                            choices=freeze_mode_choices)
    parser.add_argument('-freezeMode2', type=str,
                            choices=freeze_mode_choices)
    parser.add_argument('-freezeMode2Freq', type=str,
                            help='Do a freezeMode2 iteration every X batches.'
                                 ' X should be an integer or a string with the '
                                 'letter "e" followed by an integer. If the '
                                 'latter, every X epochs instead of batches.')

    parser.add_argument('-zSourceFine', default='policy',
                            choices=['policy', 'prior', 'speaker', 'encoder'])

    parser.add_argument('-batchSize', default=400, type=int,
                            help='Batch size (number of threads) '
                                    '(Adjust base on GPU memory)')
    parser.add_argument('-learningRate', default=4e-4, type=float,
                            help='Learning rate')
    parser.add_argument('-minLRate', default=5e-5, type=float,
                            help='Minimum learning rate')
    parser.add_argument('-dropout', default=0, type=float, help='Dropout')
    parser.add_argument('-numEpochs', default=30, type=int, help='Epochs')
    parser.add_argument('-lrDecayRate', default=0.999962372474343, type=float,
                            help='Decay for learning rate')
    parser.add_argument('-scheduler', default='none',
                            choices=['none', 'plateau'])
    parser.add_argument('-lrPatience', default=10, type=int,
                            help='Patience for plateau scheduler')
    parser.add_argument('-lrCooldown', default=0, type=int,
                            help='Cooldown for plateau scheduler')
    parser.add_argument('-predictLossCoeff', default=10, type=float,
                            help='Coefficient for feature regression loss')
    parser.add_argument('-qcycleLossCoeff', default=0, type=float,
                            help='Coefficient for question cycle consistency loss')
    parser.add_argument('-ewcLossCoeff', default=0, type=float,
                            help='Coefficient for EWC loss')

    # Other training environmnet settings
    parser.add_argument('-useGPU', action='store_true', help='Use GPU or CPU')
    parser.add_argument('-numWorkers', default=20, type=int,
                            help='Number of worker threads in dataloader')
    parser.add_argument('-useRedis', default=0, type=int,
                        help='set to 1 to store dataset in redis memory '
                             '(requires configuring redis instance)')

    #-------------------------------------------------------------------------
    # Evaluation params
    parser.add_argument('-beamSize', default=1, type=int,
                            help='Beam width for beam-search sampling')
    parser.add_argument('-evalModeList', default=[], nargs='+',
                            help='What task should the evaluator perform?',
                            choices=['ABotRank', 'QBotRank', 'QABotsRank', 'dialog'])
    parser.add_argument('-evalSplit', default='val1',
                            help='Which split to evaluate with:\n'
                                 'val1 and val2 are partitions of vqa val\n'
                                 '  val1 is meant to be a validation set\n'
                                 '  val2 is meant to act like a test set',
                            choices=['train', 'val', 'test', 'val1', 'val2'])
    parser.add_argument('-evalInference', default='sample,sample',
                            choices=['sample,sample', 'sample,argmax', 'argmax,sample', 'argmax,argmax'],
                            help='Inference for z and the decoder in the format\n'
                                 '  <z inference>,<decoder inference>\n'
                                 'where each inference parameter is one of '
                                 'sample and argmax')
    parser.add_argument('-evalTitle', default='eval',
                            help='If generating a plot, include this in the title')
    parser.add_argument('-evalLimit', type=int,
                            help='Limit evaluation to first N examples. (use '
                                 'all by default)')
    parser.add_argument('-languageModel',
                            help='Checkpoint for a questioner language model.')

    #-------------------------------------------------------------------------
    # Visualization params
    parser.add_argument('-visMode', default='latent',
                            choices=['latent', 'interpolate', 'dialog', 'mturk'],
                            help='latent - visualize latent variable samples \n'
                                 'interpolate - visualize interpolations on z \n'
                                 'dialog - visualize dialog rollouts \n'
                                 'mturk - generate data for mturk hits')
    parser.add_argument('-batchIndex', default=0, type=int,
                            help='Which batch to visualize (0-based index)')

    #-------------------------------------------------------------------------

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))

    if parsed['saveName']:
        # Custom save file path
        parsed['savePath'] = os.path.join(parsed['savePath'],
                                          parsed['saveName'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
        parsed['savePath'] = os.path.join(parsed['savePath'], timeStamp)
        parsed['savePath'] += '_{:0>6d}'.format(random.randint(0, 10e6))

    return parsed


def model_params(params):
    return {
        'latentSize': params['latentSize'],
        'rnnHiddenSize': params['rnnHiddenSize'],
        'wordDropoutRate': params['wordDropoutRate'],
        'embedSize': params['embedSize'],
        'num_embeddings': params['num_embeddings'],
        'num_vars': params['num_vars'],
        'varType': params['varType'],
        'decoderVar': params['decoderVar'],
        'queryType': params.get('queryType', 'dialog_only'),
        'speakerType': params.get('speakerType', 'two_part'),
        'imgEmbedSize': params['imgEmbedSize'],
        'ansEmbedSize': params['ansEmbedSize'],
        'numLayers': params['numLayers'],
        'temp': params['temp'],
        'tempAnnealRate': params['tempAnnealRate'],
        'tempMin': params['tempMin'],
        'softStartIters': params['softStartIters'],
        'imgFeatureSize': params['imgFeatureSize'],
        'dropout': params['dropout'],
        'embedSize': params['embedSize'],
        'conditionEncoder': params['conditionEncoder'],
        'freezeMode' : params['freezeMode'],
        # abot
        'ansObjective': params['ansObjective'],
        'abotTrackAttention': params.get('qcycleLossCoeff', 0) != 0,
    }


def data_params(params):
    return {
        # VQA
        'inputQues': params['inputQues'],
        'inputJson': params['inputJson'],
        'splitInfo': params.get('splitInfo'),
        'randQues': params['trainMode'] == 'pre-abot',
        # CUB
        'cubInfo': params.get('cubInfo'),
        # AWA
        'awaInfo': params.get('awaInfo'),
        # used for both
        'inputImg': params['inputImg'],
        'poolInfo': params.get('poolInfo'),
        'useRedis': params.get('useRedis', 0),
        'poolType': params['poolType'],
        'poolSize': params['poolSize'],
    }

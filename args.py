import argparse
import torch
from model.hyperparameters.best_parameters import encoder, task_weighting, decoder
from pathlib import Path
from utils import load_json
from datetime import datetime
import random
import sys


def add_tune_params(parser):
    parser.add_argument('--grace_period', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--gpus_per_trial', type=int, default=1)
    parser.add_argument('--cpus_per_trial', type=int, default=2)
    return parser

def add_lstm_params(parser):
    parser.add_argument('--h_dim_lstm', type=int, default=128)
    parser.add_argument('--num_layers_lstm', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.05)
    return parser

def add_transformer_params(parser):
    parser.add_argument('--h_dim_transformer', type=int, default=128)
    parser.add_argument('--num_layers_transformer', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--transformer_dropout', type=float, default=0.3)
    return parser

def add_task_weighting_params(parser):
    # note that the relative weighting between these will also depend on whether class weighting is used
    # the class weight argument will upweight the binary task and downweight the classification task
    parser.add_argument('--duration_weight', type=float, default=0.5)
    parser.add_argument('--binary_weight', type=float, default=1)
    parser.add_argument('--categorical_weight', type=float, default=0.2)
    parser.add_argument('--ts_reconstruction_weight', type=float, default=0.1)
    parser.add_argument('--ts_forecasting_weight', type=float, default=0.1)
    parser.add_argument('--binary_reconstruction_weight', type=float, default=0.002)  # it was found that this task was very easy and therefore takes over in the representation space
    parser.add_argument('--continuous_reconstruction_weight', type=float, default=0.1)
    return parser

def add_tpc_params(parser):
    parser.add_argument('--num_layers_tpc', default=6, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--no_temp_kernels', default=6, type=int)
    parser.add_argument('--point_size', default=14, type=int)
    parser.add_argument('--temp_dropout_rate', default=0.05, type=float)
    parser.add_argument('--last_linear_size', default=16, type=int)
    parser.add_argument('--encoding_dim_tpc', default=128, type=int)
    parser.add_argument('--momentum', default=0.1, type=float)
    return parser

def initialise_arguments():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--config_file', type=str, default='paths.json',
                        help='Config file path - command line arguments will override those in the file.')
    parser.add_argument('--read_best', action='store_true')
    parser.add_argument('--seed', type=int, default=random.randint(0,10000))
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpus', type=int, default=-1, help='number of available GPUs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=20)

    # paths
    parser.add_argument('--version', type=str, help='version tag')
    parser.add_argument('--data_dir', type=str, help='path of dir storing raw data')
    parser.add_argument('--log_path', type=str, help='path to store model')
    parser.add_argument('--load_encoder', type=str, help='path to load encoder from, within the logs folder')

    # training
    parser.add_argument('--test', action='store_true', help='skip training phase')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau'], default='plateau')
    parser.add_argument('--class_weights', action='store_true')  # note that this will affect the task weighting (see comments above)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=8)

    # pytorch lightning trainer stuff
    parser.add_argument('--auto_lr', action='store_true')
    parser.add_argument('--auto_bsz', action='store_true')

    # data
    parser.add_argument('--F', type=int, default=31)
    parser.add_argument('--no_flat_features', type=int, default=35)
    parser.add_argument('--max_seq_len', type=int, default=21*24)

    # tasks
    parser.add_argument('--duration_prediction_tasks', nargs='+', help='Enter the tasks. ', default=['lengthofstay_from_ventstart', 'actualventduration'])
    parser.add_argument('--binary_prediction_tasks', nargs='+', help='Enter the tasks. ', default=['diedinicu', 'tracheostomy'])
    parser.add_argument('--categorical_prediction_tasks',  nargs='+', help='Enter the tasks. ', default=[])#'destination', 'diagnosissubgroup'])
    parser.add_argument('--last_timestep_reconstruction_tasks', nargs='+', help='Enter the tasks. ', default=['all'])
    parser.add_argument('--forecast_reconstruction_tasks', nargs='+', help='Enter the tasks. ', default=['all'])
    parser.add_argument('--binary_reconstruction_tasks', nargs='+', help='Enter the tasks. ', default=['urgency', 'gender'])
    parser.add_argument('--continuous_reconstruction_tasks', nargs='+', help='Enter the tasks. ', default=['agegroup', 'weightgroup', 'heightgroup'])

    # hyperparameters
    parser.add_argument('--encoder', type=str, choices=['lstm', 'tpc', 'transformer'], default='tpc')
    parser.add_argument('--h_dim_decoder', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--fc_activate_fn', type=str, default='relu')
    parser.add_argument('--embedding_activate_fn', type=str, default='tanh')
    parser.add_argument('--variational', action='store_true')
    parser = add_lstm_params(parser)
    parser = add_tpc_params(parser)
    parser = add_transformer_params(parser)
    parser = add_task_weighting_params(parser)

    return parser


def add_best_params(config):
    """
    read best set of hyperparams
    """
    for key, value in encoder[config['encoder']].items():
        config[key] = value
    config['version'] = 'best_{}'.format(config['encoder']) if config['version'] is None else config['version']
    print('*** using best values for {} encoder params'.format(config['encoder']), [p for p in encoder[config['encoder']]])
    for params in task_weighting, decoder:
        for key, value in params.items():
            config[key] = value
    print('*** and these', [p for p in task_weighting], [p for p in decoder])

    return config


def get_version_name(config):
    """
    return str for model version name
    """

    if config['version'] is None:
        # first about the model
        version = config['encoder'].upper()
        # finally about training
        now = datetime.now()
        date_time = now.strftime('_%m%d_%H:%M:%S')
        version += date_time
        config['version'] = version

    return config


def read_params_from_file(arg_dict, overwrite=False):
    """
    Read params defined in config_file (paths.py by default.)
    """
    if '/' not in arg_dict['config_file']:
        config_path = Path(sys.path[0]) / arg_dict['config_file']
    else:
        config_path = Path(arg_dict['config_file'])

    data = load_json(config_path)
    arg_dict.pop('config_file')

    if not overwrite:
        for key, value in data.items():
            if isinstance(value, list) and (key in arg_dict):
                for v in value:
                    arg_dict[key].append(v)
            elif (key not in arg_dict) or (arg_dict[key] is None):
                arg_dict[key] = value
    else:
        for key, value in data.items():
            arg_dict[key] = value

    return arg_dict


def add_configs(config):
    """
    add in additional configs
    """
    config = vars(config)

    # training details
    if config['cpu']:
        num_gpus = 0
        config['gpus'] = None
        config['gpus_per_trial'] = 0
    else:
        if config['gpus'] is not None:
            num_gpus = torch.cuda.device_count() if config['gpus'] == -1 else config['gpus']
            if num_gpus > 0:
                config['batch_size'] *= num_gpus
                config['num_workers'] *= num_gpus
        else:
            num_gpus = 0
    config['num_gpus'] = num_gpus
    config['multi_gpu'] = num_gpus > 1

    if 'config_file' in config:
        read_params_from_file(config)

    get_version_name(config)

    if config['read_best']:
        add_best_params(config)

    # slightly hacky way to make the new version allowing for ablation studies to be compatible with old versions
    try:
        config['ts_forecasting_weight']
    except KeyError:
        config['ts_forecasting_weight'] = config['ts_reconstruction_weight']

    return config
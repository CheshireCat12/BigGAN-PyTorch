''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import os
from collections import defaultdict


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import utils


import pdb

def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                        else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    G = model.Generator(**config).cuda()
    utils.count_parameters(G)

    D = None
    # Load weights
    print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    utils.load_weights(G if not (config['use_ema']) else None, D, state_dict,
                        config['weights_root'], experiment_name, config['load_weights'],
                        G if config['ema'] and config['use_ema'] else None,
                        strict=False, load_optim=False)
    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                                device=device, fp16=config['G_fp16'],
                                z_var=config['z_var'])

    print('Putting G in eval mode..')
    G.eval()

    return functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)


def get_sampler():
    config = {'dataset': 'C100', 'augment': False, 'num_workers': 8, 'pin_memory': True,
    'shuffle': True, 'load_in_mem': False, 'use_multiepoch_sampler': False,
    'model': 'BigGANmh', 'G_param': 'SN', 'D_param': 'SN', 'G_ch': 64,
    'D_ch': 64, 'G_depth': 1, 'D_depth': 1, 'D_wide': True, 'G_shared': False,
    'shared_dim': 0, 'dim_z': 128, 'z_var': 1.0, 'hier': False,
    'cross_replica': False, 'mybn': False, 'G_nl': 'relu', 'D_nl': 'relu',
    'G_attn': '0', 'D_attn': '64', 'norm_style': 'bn', 'mh_csc_loss': False,
    'resampling': False, 'mh_loss': True, 'use_unlabeled_data': False,
    'bottom_width': 4, 'ignore_projection_discriminator': False, 'fm_loss': False,
    'mh_loss_weight': 0.05, 'mh_fmloss_weight': 1.0, 'seed': 0, 'G_init': 'N02',
    'D_init': 'ortho', 'skip_init': False, 'G_lr': 5e-05, 'D_lr': 0.0002,
    'G_B1': 0.0, 'D_B1': 0.0, 'G_B2': 0.999, 'D_B2': 0.999, 'batch_size': 50,
    'G_batch_size': 256, 'num_G_accumulations': 1, 'num_D_steps': 2,
    'num_D_accumulations': 1, 'split_D': False, 'num_epochs': 500,
    'parallel': True, 'G_fp16': False, 'D_fp16': False, 'D_mixed_precision': False,
    'G_mixed_precision': False, 'accumulate_stats': False, 'num_standing_accumulations': 16,
    'G_eval_mode': False, 'save_every': 2000, 'num_save_copies': 2, 'num_best_copies': 2,
    'which_best': 'IS', 'no_fid': False, 'test_every': 5000, 'num_inception_images': 50000,
    'hashname': False, 'base_root': '', 'data_root': './data',
    'weights_root': '/content/BigGAN-PyTorch/BigGAN/cifartest',
    'logs_root': '/content/BigGAN-PyTorch/BigGAN/cifartest',
    'samples_root': '/content/BigGAN-PyTorch/BigGAN/cifartest',
    'pbar': 'mine', 'name_suffix': '', 'experiment_name': 'c100_mh_p05',
    'config_from_name': False, 'historical_save_every': 1000000000.0,
    'ema': True, 'ema_decay': 0.9999, 'use_ema': True, 'ema_start': 1000,
    'adam_eps': 1e-08, 'BN_eps': 1e-05, 'SN_eps': 1e-08, 'num_G_SVs': 1,
    'num_D_SVs': 1, 'num_G_SV_itrs': 1, 'num_D_SV_itrs': 1, 'G_ortho': 0.0,
    'D_ortho': 0.0, 'toggle_grads': True, 'which_train_fn': 'GAN',
    'load_weights': '050000', 'resume': False, 'logstyle': '%3.3e',
    'log_G_spectra': False, 'log_D_spectra': False, 'sv_log_interval': 10,
    'sample_npz': False, 'sample_num_npz': 50000, 'sample_sheets': False,
    'sample_interps': False, 'sample_sheet_folder_num': -1, 'sample_random': False,
    'sample_trunc_curves': '', 'sample_inception_metrics': False, 'sample_np_mem': False,
    'official_IS': False, 'official_FID': False, 'overwrite': False, 'dataset_is_fid': '',
    'sample_multiple': False, 'get_test_error': False, 'get_train_error': False,
    'get_self_error': False, 'get_generator_error': False, 'sample_num_error': 10000}

    return run(config)

if __name__ == '__main__':
  main()
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
# import inception_utils
# import utils
# import losses

# import inception as iscore
# import fid

import pdb

def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'],
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]

  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
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

  # In some cases we need to load D
  # if True or config['get_test_error'] or config['get_train_error'] or config['get_self_error']or config['get_generator_error']:
  #   disc_config = config.copy()
  #   if config['mh_csc_loss'] or config['mh_loss']:
  #     disc_config['output_dim'] = disc_config['n_classes'] + 1
  #   D = model.Discriminator(**disc_config).to(device)

  #   def get_n_correct_from_D(x, y):
  #     """Gets the "classifications" from D.

  #     y: the correct labels

  #     In the case of projection discrimination we have to pass in all the labels
  #     as conditionings to get the class specific affinity.
  #     """
  #     x = x.to(device)
  #     if config['model'] == 'BigGAN': # projection discrimination case
  #       if not config['get_self_error']:
  #         y = y.to(device)
  #       yhat = D(x,y)
  #       for i in range(1,config['n_classes']):
  #         yhat_ = D(x,((y+i) % config['n_classes']))
  #         yhat = torch.cat([yhat,yhat_],1)
  #       preds_ = yhat.data.max(1)[1].cpu()
  #       return preds_.eq(0).cpu().sum()
  #     else: # the mh gan case
  #       if not config['get_self_error']:
  #         y = y.to(device)
  #       yhat = D(x)
  #       preds_ = yhat[:,:config['n_classes']].data.max(1)[1]
  #       return preds_.eq(y.data).cpu().sum()

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

  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))

  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
  brief_expt_name = config['experiment_name'][-30:]

  # load results dict always
  HIST_FNAME = 'scoring_hist.npy'
  def load_or_make_hist(d):
    """make/load history files in each
    """
    if not os.path.isdir(d):
      raise Exception('%s is not a valid directory' % d)
    f = os.path.join(d, HIST_FNAME)
    if os.path.isfile(f):
      return np.load(f, allow_pickle=True).item()
    else:
      return defaultdict(dict)
  hist_dir = os.path.join(config['weights_root'], config['experiment_name'])
  hist = load_or_make_hist(hist_dir)

  # Sample random sheet
  if config['sample_random']:
    print('Preparing random sample sheet...')
    images, labels = sample()
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/%s___.jpg' % (config['samples_root'], experiment_name, config['load_weights']),
                                 nrow=int(G_batch_size**0.5),
                                 normalize=True)



def main():
  # parse command line and run
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
  if config['sample_multiple']:
    suffixes = config['load_weights'].split(',')
    for suffix in suffixes:
      config['load_weights'] = suffix
      run(config)
  else:
    run(config)

if __name__ == '__main__':
  main()
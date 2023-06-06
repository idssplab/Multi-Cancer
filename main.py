import argparse
import collections
import pandas as pd
from mode import *
from utils import set_random_seed
import logging
from parse_config import ConfigParser

SEED = 1126
set_random_seed(SEED)

def main(config):
    logging.info('Using Random Seed {} for this experiment'.format(SEED))
    if config['mode'] == 'multi cross validation':
        cross_validation_results = multi_cross_validation(config)
        for project_id in cross_validation_results:
            cross_validation_result = cross_validation_results[project_id]

            logging.info(f'{project_id} Cross Validation Result')
            for key, value in cross_validation_result.items():
                    logging.info('{:40s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))

    elif config['mode'] == 'multi cross validation bootstrap':
        cross_validation_bootstrap_results = multi_cross_validation_bootstrap(config)
        cross_validation_results = cross_validation_bootstrap_results['cross_validation']
        bootstrap_results = cross_validation_bootstrap_results['bootstrap']

        for project_id in cross_validation_results:
            cross_validation_result = cross_validation_results[project_id]

            logging.info(f'{project_id} Cross Validation Result')
            for key, value in cross_validation_result.items():
                    logging.info('{:40s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))

        for project_id in bootstrap_results:
            bootstrap_result = bootstrap_results[project_id]

            logging.info(f'{project_id} Bootstrapping Result')
            for key, value in bootstrap_result.items():
                    logging.info('{:40s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))
    elif config['mode'] == 'multi train bootstrap':
        train_bootstrap_results = multi_train_bootstrap(config)
        for project_id in train_bootstrap_results:
            train_bootstrap_result = train_bootstrap_results[project_id]

            logging.info(f'{project_id} Train Bootstrapping Result')
            for key, value in train_bootstrap_result.items():
                    logging.info('{:40s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))

    else:
        raise KeyError(f'Mode {config["mode"]} not supported')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Main')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable')
    args.add_argument('--run_id', default=None, type=str, help='')

    # Custom CLI options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

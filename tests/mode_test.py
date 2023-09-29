import argparse
from parse_config import ConfigParser
from mode import multi_cross_validation_bootstrap, multi_train_bootstrap
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)


def main(config):
    if config['mode'] == 'multi cross validation bootstrap':
        cross_validation_bootstrap_results = multi_cross_validation_bootstrap(config)
        print(cross_validation_bootstrap_results)
    elif config['mode'] == 'multi train bootstrap':
        train_bootstrap_results = multi_train_bootstrap(config)
        print(train_bootstrap_results)
    else:
        raise KeyError(f'Mode {config["mode"]} not supported')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Mode Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, run_id='mode_test')
    main(config)

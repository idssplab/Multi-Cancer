import argparse
import model as module_arch
from parse_config import ConfigParser
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

def main(config):
    Model = config.init_obj('arch', module_arch)

    print(Model)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, run_id='model_test')
    main(config)

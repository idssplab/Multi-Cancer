import argparse
import pandas as pd
import dataset as module_dataset
import datasets_manager as module_datasets_manager
from parse_config import ConfigParser
from runner import Tester
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

def main(config):
    Datasets = {project_id: config.init_obj(
            f'datasets.{project_id}',
            module_dataset
        )
        for project_id in config['datasets']
    }

    Datasets_Manager = getattr(
        module_datasets_manager,
        config['datasets_manager.type']
    )(datasets=Datasets, config=config)

    TCGA_BRCA_Data_Loaders = Datasets_Manager['TCGA-BRCA']['dataloaders']
    test_data_loader = TCGA_BRCA_Data_Loaders['test']

    tester = Tester(
        config=config,
        test_data_loader=test_data_loader
    )

    tester.run()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, run_id='tester_test')
    main(config)

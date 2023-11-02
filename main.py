import argparse
import collections
import logging

from mode import multi_cross_validation_bootstrap, multi_train_bootstrap
from parse_config import ConfigParser
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)


def main():
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

    # NOTE: Override n_genes in config file for multi-task graph neural network.
    #override_n_genes(config)

    logging.info(f'Using Random Seed {SEED} for this experiment')

    if config['mode'] == 'multi cross validation bootstrap':
        results = multi_cross_validation_bootstrap(config)
    elif config['mode'] == 'multi train bootstrap':
        results = {'train_boostrap': multi_train_bootstrap(config)}
    else:
        raise KeyError(f'Invalid Mode {config["mode"]}')

    show_project_result(results.get('cross_validation', {}), 'Cross Validation Result')
    show_project_result(results.get('bootstrap', {}), 'Bootstrap Result')
    show_project_result(results.get('train_boostrap', {}), 'Train Bootstrapping Result')


def show_project_result(results: dict[str, dict], postfix: str):
    for project_id in results:
        project_result = results[project_id]
        logging.info(f'{project_id} {postfix}')
        for key, value in project_result.items():
            logging.info('{:40s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))


def override_n_genes(config: ConfigParser):
    """Override n_genes in config file.

    Args:
        config (ConfigParser): config object with muteble data structure.
    """
    genes = config['datasets.TCGA_BLC.args.chosen_features.gene_ids'] if 'TCGA_BLC' in config['datasets'] else None
    try:
        if genes is None:   # TCGA_Project_Dataset.
            n_genes = config['models.Feature_Extractor.args.n_genes']
        elif isinstance(genes, list):
            n_genes = len(genes)
        elif isinstance(genes, dict):
            all_selected_genes = set()
            for genes in genes.values():
                all_selected_genes.update(genes)
            n_genes = len(all_selected_genes)
        config['models']['Feature_Extractor']['args']['n_genes'] = n_genes
    except KeyError:        # No Feature_Extractor in config file or no n_genes in Feature_Extractor args.
        pass


if __name__ == '__main__':
    main()

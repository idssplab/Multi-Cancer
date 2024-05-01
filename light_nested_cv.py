import argparse
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from dataset import TCGA_Program_Dataset
from datasets_manager import TCGA_Balanced_Datasets_Manager, TCGA_Datasets_Manager
from lit_models import LitFullModel
from model import Classifier, Feature_Extractor, Graph_And_Clinical_Feature_Extractor, Task_Classifier
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging

SEED = 1126
set_random_seed(SEED)

import numpy as np
from sklearn.model_selection import KFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    override_n_genes(config)
    config_name = Path(args.config).stem
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    
    data = {'TCGA_BLC': TCGA_Program_Dataset(**config['datasets'])}
    if 'TCGA_Balanced_Datasets_Manager' == config['datasets_manager']['type']:
        manager = TCGA_Balanced_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))
    else:
        manager = TCGA_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))


    n_splits_outer = 5
    n_splits_inner = 5
    #outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=SEED)
    outer_cv = manager.get_kfold_samplers(data, n_splits_outer)

    outer_results = []

    for train_idx, test_idx in outer_cv.split(data['TCGA_BLC']):
        train_data, test_data = data['TCGA_BLC'][train_idx], data['TCGA_BLC'][test_idx]

        # Inner loop for model selection and hyperparameter tuning
        inner_cv = manager.get_kfold_samplers(data, n_splits_inner)
        #inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=SEED)
        inner_results = []

        for inner_train_idx, inner_val_idx in inner_cv.split(train_data):
            inner_train_data, inner_val_data = train_data[inner_train_idx], train_data[inner_val_idx]

            # Model training
            models, optimizers = create_models_and_optimizers(config)
            lit_model = LitFullModel(models, optimizers, config)
            trainer = pl.Trainer(default_root_dir=log_path, max_epochs=config['max_epochs'])
            trainer.fit(lit_model, train_dataloaders=inner_train_data, val_dataloaders=inner_val_data)
            
            # Validate the model
            validation_result = trainer.test(lit_model, dataloaders=inner_val_data, verbose=False)
            inner_results.append(validation_result)

        # Select the best model based on inner loop
        best_model_idx = np.argmax([result['val_acc'] for result in inner_results])  # Example metric
        best_model = models[best_model_idx]

        # Test the best model
        final_trainer = pl.Trainer(default_root_dir=log_path, max_epochs=config['max_epochs'])
        test_result = final_trainer.test(best_model, dataloaders=test_data, verbose=False)
        outer_results.append(test_result)

    # Aggregate and report results from outer loop
    outer_results_df = pd.DataFrame.from_records(outer_results)
    for key, value in outer_results_df.describe().loc[['mean', 'std']].to_dict().items():
        logger.info(f'| {key.ljust(10).upper()} | {value["mean"]:.5f} ± {value["std"]:.5f} |')






def create_models_and_optimizers(config: dict):
    models: dict[str, torch.nn.Module] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}

    # Setup models. Do not use getattr() for better IDE support.
    for model_name, kargs in config['models'].items():
        if model_name == 'Graph_And_Clinical_Feature_Extractor':
            models['feat_ext'] = Graph_And_Clinical_Feature_Extractor(**kargs)
        elif model_name == 'Feature_Extractor':
            models['feat_ext'] = Feature_Extractor(**kargs)
        elif model_name == 'Task_Classifier':
            models['clf'] = Task_Classifier(**kargs)
        elif model_name == 'Classifier':
            models['clf'] = Classifier(**kargs)
        else:
            raise ValueError(f'Unknown model type: {model_name}')

    # Setup optimizers. If the key is 'all', the optimizer will be applied to all models.
    for key, optim_dict in config['optimizers'].items():
        opt_name = next(iter(optim_dict))
        if key == 'all':
            params = [param for model in models.values() for param in model.parameters()]
            optimizers[key] = getattr(torch.optim, opt_name)(params, **optim_dict[opt_name])
        else:
            optimizers[key] = getattr(torch.optim, opt_name)(models[key].parameters(), **optim_dict[opt_name])

    # Add models' structure to config for logging. TODO: Prettify.
    for model_name, torch_model in models.items():
        config[f'model.{model_name}'] = str(torch_model)
    return models, optimizers


if __name__ == '__main__':
    main()

import argparse
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger
from tqdm import tqdm
from dataset import TCGA_Program_Dataset
from datasets_manager import TCGA_Balanced_Datasets_Manager, TCGA_Datasets_Manager
from lit_models import LitFullModel
from model import Classifier, Feature_Extractor, Graph_And_Clinical_Feature_Extractor, Task_Classifier, Genomic_Separate_Feature_Extractor, Clinical_Separate_Feature_Extractor
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging
from lightningDM import DataModule
from sklearn.model_selection import KFold,  StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid

SEED = 1126
set_random_seed(SEED)


def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    override_n_genes(config)       # For multi-task graph models.
    config['csv_logger'] = True if 'csv_logger' in config and config['csv_logger'] else False                                             
    config_name = Path(args.config).stem

     # Setup logging.
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    get_logger('lightning.pytorch.accelerators.cuda', log_level='WARNING')      # Disable cuda logging.
    filterwarnings('ignore', r'.*Skipping val loop.*')                          # Disable val loop warning.
  
    if config['cross_validation'] == True:
        nested_cross_validation(logger, log_path, config)
    # Bootstrap with the final model.
    # if config['bootstrap_repeats'] > 0:
    #     bootstrap_with_final_model(logger, log_path, config)

      


def nested_cross_validation(logger, log_path, config: dict):
    data = DataModule(**config['datasets'], no_validation_set=True)
    data.setup()

    n_folds = config['n_folds']  # Number of folds for both outer and inner cross-validation
    train_dataset = data.train_dataloader().dataset
    targets = train_dataset.get_targets()  # Assuming your dataset has a get_targets() method
    batch_size = config['datasets']['batch_size']
    num_workers = config['datasets']['num_workers']

    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'lr2': [5e-3, 5e-4, 5e-5],
        'batch_size': [32, 64, 128],
        'momentum': [0.9, 0.95],
        'max_epochs': [40, 50, 60],
    }


    # Outer cross-validation setup
    outer_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
    outer_results = []

    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(train_dataset, targets)):
        print(f"Outer fold {outer_fold + 1}/{n_folds}")

        outer_train_targets = [targets[idx] for idx in outer_train_idx]
        outer_train_subset = torch.utils.data.Subset(train_dataset, outer_train_idx)
        test_subset = torch.utils.data.Subset(train_dataset, outer_test_idx)

        # Inner cross-validation setup
        inner_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        best_hyperparams = None
        best_val_score = float('-inf')

        # Iterate over each fold for the inner cross-validation

        for params in ParameterGrid(param_grid):
            inner_val_scores = []
            for inner_fold, (train_idx, val_idx) in enumerate(inner_kfold.split(outer_train_subset.indices, outer_train_targets)):
                print(f"Training outer fold {outer_fold + 1}/{n_folds}")
                print(f"Training inner fold {inner_fold + 1}/{n_folds}")
                # print the parameters to be used
                print(params)

                train_subset = torch.utils.data.Subset(train_dataset, [outer_train_idx[i] for i in train_idx])
                val_subset = torch.utils.data.Subset(train_dataset, [outer_train_idx[i] for i in val_idx])

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=data.collate_fn, drop_last=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data.collate_fn, drop_last=False)

                config['optimizers']['feat_ext']['AdamW']['lr'] = params['lr']
                config['optimizers']['clf']['SGD']['lr'] = params['lr2']
                config['datasets']['batch_size'] = params['batch_size']
                config['optimizers']['clf']['SGD']['momentum'] = params['momentum']
                config['max_epochs'] = params['max_epochs']
                # config_add_subdict_key(config, 'optimizers.feat_ext.AdamW.lr', params['lr'])
                # config_add_subdict_key(config, 'optimizers.clf.SGD.lr', params['lr2'])
                # config_add_subdict_key(config, 'datasets', 'batch_size', params['batch_size'])
                # config_add_subdict_key(config, 'optimizers.feat_ext.SGD', 'momentum', params['momentum'])
                # config_add_subdict_key(config, 'max_epochs', params['max_epochs'])

                models, optimizers = create_models_and_optimizers(config)
                lit_model = LitFullModel(models, optimizers, config)
                trainer = pl.Trainer(
                    default_root_dir=log_path,
                    max_epochs=config['max_epochs'],
                    log_every_n_steps=1,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    accelerator='gpu',
                    check_val_every_n_epoch=1,
                    logger=CSVLogger(save_dir=log_path) if config['csv_logger'] else True,
                )

                # Train the model on the inner loop
                # Train the model on the inner loop
                trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                val_score = trainer.validate(lit_model, dataloaders=val_loader, verbose=False)[0]
                inner_val_scores.append(sum(val_score.values()) / len(val_score))

            avg_val_score = sum(inner_val_scores) / len(inner_val_scores)
            print(f"Average validation score for hyperparameters {params}: {avg_val_score}")
            if avg_val_score > best_val_score:
                best_val_score = avg_val_score
                best_hyperparams = params
                print(f"New best hyperparameters: {best_hyperparams}")

        print(f"Best hyperparameters for outer fold {outer_fold + 1}: {best_hyperparams}")

        # Train the model on the outer loop with the best hyperparameters

        config['optimizers']['feat_ext']['AdamW']['lr'] = best_hyperparams['lr']
        config['optimizers']['clf']['SGD']['lr'] = best_hyperparams['lr2']
        config['datasets']['batch_size'] = best_hyperparams['batch_size']
        config['optimizers']['clf']['SGD']['momentum'] = best_hyperparams['momentum']
        config['max_epochs'] = best_hyperparams['max_epochs']



        models, optimizers = create_models_and_optimizers(config)
        lit_model = LitFullModel(models, optimizers, config)

        outer_train_loader = DataLoader(outer_train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=data.collate_fn, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data.collate_fn, drop_last=False)

        trainer = pl.Trainer(
            default_root_dir=log_path,
            max_epochs=config['max_epochs'],
            log_every_n_steps=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            accelerator='gpu',
            check_val_every_n_epoch=1,
            logger=CSVLogger(save_dir=log_path) if config['csv_logger'] else True,
        )

        # Train and test the model on the outer loop
        trainer.fit(lit_model, train_dataloaders=outer_train_loader)
        outer_test_result = trainer.test(lit_model, dataloaders=test_loader, verbose=False)[0]
        outer_results.append(outer_test_result)

    # Print and save outer loop results
    logger.info(f'{"-" * 25} Outer Loop Test Results {"-" * 25}')
    if outer_results:
        df_outer_results = pd.DataFrame.from_records(outer_results)
        logger.info('\n' + results_to_markdown_table(df_outer_results, config, 'test'))

    # Save final model
    trainer.save_checkpoint(f'{log_path}/final_nested_cv_model.ckpt')

  



    


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
        elif model_name == 'Genomic_Separate_Feature_Extractor':
            models['feat_ext'] = Genomic_Separate_Feature_Extractor(**kargs)
        elif model_name == 'Clinical_Separate_Feature_Extractor':
            models['feat_ext'] = Clinical_Separate_Feature_Extractor(**kargs)
        
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

def results_to_markdown_table(df_results: pd.DataFrame, config: dict, mode: str) -> str:
    describe = df_results.describe()
    metrics = list(dict.fromkeys([col.split('_')[0] for col in describe.columns if 'loss' not in col]))
    losses = [col for col in describe.columns if 'loss' in col]

    table = pd.DataFrame(columns=[mode] + metrics + losses)
    table[mode] = config['datasets']['project_ids'] + ['all'] if losses else ''
    table.set_index(mode, inplace=True)

    for metric in metrics:
        for project_id, project in enumerate(config['datasets']['project_ids']):
            col = f'{metric}_{project_id}'
            try:
                assert col in describe.columns, f'{col} not in {describe.columns} when summarizing {mode} results.'
            except AssertionError:
                # remove the 0 from the column names
                describe.columns = [col.replace('.0', '') for col in describe.columns]
            table.loc[project, metric] = f'{describe.loc["mean", col]:.5f} ± {describe.loc["std", col]:.5f}'
    for loss in losses:
        assert loss in describe.columns, f'{loss} not in {describe.columns} when summarizing {mode} results.'
        table.loc['all', loss] = f'{describe.loc["mean", loss]:.5f} ± {describe.loc["std", loss]:.5f}'
    return table.to_markdown(tablefmt='pipe', stralign='right', numalign='right')


if __name__ == '__main__':
    main()

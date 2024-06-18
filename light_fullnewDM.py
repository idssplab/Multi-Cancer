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
from model import Classifier, Feature_Extractor, Task_Classifier
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging
from lightningDM import DataModule
from sklearn.model_selection import KFold,  StratifiedKFold
from torch.utils.data import DataLoader

CUDA_LAUNCH_BLOCKING=1
SEED = 1126
set_random_seed(SEED)


def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = override_n_genes(config)     
    config['csv_logger'] = True if 'csv_logger' in config and config['csv_logger'] else False                                             
    config_name = Path(args.config).stem

     # Setup logging.
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    get_logger('lightning.pytorch.accelerators.cuda', log_level='WARNING')      # Disable cuda logging.
    filterwarnings('ignore', r'.*Skipping val loop.*')                          # Disable val loop warning.
  
    if config['cross_validation'] == True:
        cross_validation(logger, log_path, config)
    # Bootstrap with the final model.
    if config['bootstrap_repeats'] > 0:
        bootstrap_with_final_model(logger, log_path, config)

      


def cross_validation(logger, log_path, config: dict):
    
    
    data = DataModule(**config['datasets'], no_validation_set=True) 
    data.setup()
          
    n_folds = config['n_folds']
    train_dataset = data.train_dataloader().dataset 
    batch_size = config['datasets']['batch_size']
    num_workers = config['datasets']['num_workers']

    
    

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True) 
    validation_results = []
    
    print(f"Training {len(train_dataset.get_targets())} targets")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset, train_dataset.get_targets())):
        print(f"Training fold {fold+1}/{n_folds}")

       

        # Subset your dataset based on the indices for train and validation
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        
        collate_fn = data.collate_fn

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=True )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)           

           
       

        models, optimizers = create_models_and_optimizers(config)
        lit_model = LitFullModel(models, optimizers, config)
        trainer = pl.Trainer(                                              
            default_root_dir=log_path,
            max_epochs=config['max_epochs'],
            log_every_n_steps=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            accelerator='gpu',
            check_val_every_n_epoch=50,
            logger= CSVLogger(save_dir=log_path) if config['csv_logger'] else True,
        
        )

        # Train the model
        trainer.fit(lit_model, train_dataloaders=train_loader)
        if config['csv_logger']:                
               
                validation_results.append(trainer.test(lit_model, dataloaders=val_loader, verbose=False)[0]) 


    # Print validation results.        
    logger.info(f'{"-" * 25} Validation Results {"-" * 25}')
    
    if validation_results:
        df_valid_results = pd.DataFrame.from_records(validation_results)
     
        logger.info('\n' + results_to_markdown_table(df_valid_results, config, 'validation'))

    # Save CV model 
    trainer.save_checkpoint(f'{log_path}/cv_model.ckpt')

  


def bootstrap_with_final_model(logger, log_path, config: dict):

    
    
    
    models, optimizers = create_models_and_optimizers(config)
    lit_model = LitFullModel(models, optimizers, config)
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=config['max_epochs'],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=False,
        accelerator='gpu',
    )
    # Load data without validation set.
    data = DataModule(**config['datasets'], no_validation_set=True) 
    data.setup()
    train, test = data.train_dataloader(), data.test_dataloader()
    lit_model.load_from_checkpoint(f'{log_path}/cv_model.ckpt', models=models, optimizers=optimizers, config=config)
    

    # Test the final model.
    logger.info(f'{"-" * 25} Bootstrap Test Results {"-" * 25}')
    bootstrap_results = []
    for _ in tqdm(range(config['bootstrap_repeats']), desc='Bootstrapping'): 
              
        test = data.test_dataloader()
        
        bootstrap_results.append(trainer.test(lit_model, dataloaders=test, verbose=False)[0])  

    bootstrap_results = pd.DataFrame.from_records(bootstrap_results)
    logger.info('\n' + results_to_markdown_table(bootstrap_results, config, 'test'))

def create_models_and_optimizers(config: dict):
    models: dict[str, torch.nn.Module] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}

    # Setup models. Do not use getattr() for better IDE support.
    for model_name, kargs in config['models'].items():
        if model_name == 'Feature_Extractor':
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

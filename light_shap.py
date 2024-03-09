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
import shap
from captum.attr import IntegratedGradients
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import seaborn as sns

SEED = 1126
set_random_seed(SEED)


def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    override_n_genes(config)                                                    # For multi-task graph models.
    config_name = Path(args.config).stem

    # Setup logging.
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    get_logger('lightning.pytorch.accelerators.cuda', log_level='WARNING')      # Disable cuda logging.
    filterwarnings('ignore', r'.*Skipping val loop.*')                          # Disable val loop warning.

    # Create dataset manager.
    #here use torch lightning DS
    data = {'TCGA_BLC': TCGA_Program_Dataset(**config['datasets'])}
    if 'TCGA_Balanced_Datasets_Manager' == config['datasets_manager']['type']:
        manager = TCGA_Balanced_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))
    else:
        manager = TCGA_Datasets_Manager(datasets=data, config=config_add_subdict_key(config))

    valid_results = []
    # Cross validation.
    for key, values in manager['TCGA_BLC']['dataloaders'].items():
        if isinstance(key, int) and config['cross_validation']:
            pass
        #     models, optimizers = create_models_and_optimizers(config)
        #     lit_model = LitFullModel(models, optimizers, config)
        #     trainer = pl.Trainer(                                               # Create sub-folders for each fold.
        #         default_root_dir=log_path,
        #         max_epochs=config['max_epochs'],
        #         log_every_n_steps=1,
        #         enable_model_summary=False,
        #         enable_checkpointing=False,
        #     )
        #     trainer.fit(lit_model, train_dataloaders=values['train'], val_dataloaders=values['valid'])
        #     trainer.test(lit_model, dataloaders=values['valid'], verbose=False) #verbose true prints the validation results for each fold
        #     # print validation results
            
        #     valid_results.append(trainer.test(lit_model, dataloaders=values['valid'], verbose=False)[0])
               
            
            
        elif key == 'train':
            train = values
        elif key == 'test':
            test = values



    # Train the final model.
    models, optimizers = create_models_and_optimizers(config)
    lit_model = LitFullModel(models, optimizers, config)
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=config['max_epochs'],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=True,
    )
    #trainer.fit(lit_model, train_dataloaders=train)

    #save trained model
    #torch.save(lit_model.state_dict(), 'trained_model.pth')

    #load trained model
    lit_model.load_state_dict(torch.load('trained_model.pth'))    
    #get one batch of data
    batch = next(iter(train))
    (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch

    # if project_id == 0:
    #     feature_ids = ['ESR1', 'EFTUD2', 'HSPA8', 'STAU1', 'SHMT2', 'ACTB', 'GSK3B', 'YWHAB', 'UBXN6', 'PRKRA', 'BTRC', 'DDX23', 'SSR1', 'TUBA1C', 'SNIP1', 'SRSF5', 'ERBB2', 'MKI67', 'PGR', 'PLAU']
    # elif project_id == 1:  
    #     feature_ids= ['HNRNPU', 'STAU1', 'KDM1A', 'SERBP1', 'DHX9', 'EMC1', 'SSR1', 'PUM1', 'CLTC', 'PRKRA', 'KRR1', 'OCIAD1', 'CDC73', 'SLC2A1', 'HIF1A', 'PKM', 'CADM1', 'EPCAM', 'ALCAM', 'PTK7']
    # else:
    #     feature_ids= ['HNRNPL', 'HNRNPU', 'HNRNPA1', 'ZBTB2', 'SERBP1', 'RPL4', 'HNRNPK', 'HNRNPR', 'TFCP2', 'DHX9', 'RNF4', 'PUM1', 'ABCC1', 'CD44', 'ALCAM', 'ABCG2', 'ALDH1A1', 'ABCB1', 'EPCAM', 'PROM1']
    
         

    # get SHAP values from trained model   
    feat_extractor = models['feat_ext']
    genomic_feat = feat_extractor.genomic_feature_extractor
    #explanation = shap.GradientExplainer(feat_extractor, [genomic, clinical, project_id])
    explanation = shap.DeepExplainer(genomic_feat, genomic)
    #test_batch = next(iter(test))
    #(genomic, clinical, index, project_id), (over
    # all_survival, survival_time, vital_status) = test_batch
    shap_values = explanation.shap_values(genomic)
    print(shap_values)
    # plot the SHAP values
    shap.summary_plot(shap_values, genomic.numpy()) #, feature_names = feature_ids )  # Convert to numpy if 'genomic' is a tensor
    #rename features


    plt.savefig('shap_values.png')


    

    # get SHAP values for all the training data, going through all the batches
    all_shap_values = []
    for batch in train:
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        shap_values = explanation.shap_values(genomic)
        all_shap_values.append(shap_values)
    all_shap_values = np.concatenate(all_shap_values, axis=1)
    shap_values_df = pd.DataFrame(all_shap_values)
    correlation_matrix = all_shap_values.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation matrix of SHAP values')
    plt.savefig('SHAPcorrelation_matrix.png')
        







    # # Test the final model.
    # bootstrap_results = []
    # for _ in tqdm(range(config['bootstrap_repeats']), desc='Bootstrapping'):
    #     bootstrap_results.append(trainer.test(lit_model, dataloaders=test, verbose=False)[0])
    # bootstrap_results = pd.DataFrame.from_records(bootstrap_results)
    # for key, value in bootstrap_results.describe().loc[['mean', 'std']].to_dict().items():
    #     logger.info(f'| {key.ljust(10).upper()} | {value["mean"]:.5f} ± {value["std"]:.5f} |')


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

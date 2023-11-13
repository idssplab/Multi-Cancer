import pytest
import pandas as pd
from torch.utils.data import DataLoader
from external_lightningdatamodule import ExternalDataModule
import numpy as np
import argparse
from utils import config_add_subdict_key, get_logger, override_n_genes, set_random_seed, setup_logging
import yaml
from tqdm import tqdm
import argparse
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings


SEED = 1126
set_random_seed(SEED)

@pytest.fixture
def data_module():
    return ExternalDataModule(
        project_id=['SCLC'],
        data_dir='Data/sclc_ucologne_2015',
        cache_directory='Cache/SCLC',
        batch_size=128,
        num_workers=4,
        chosen_features={
            'gene_ids': {'TP53', 'RB1', 'TTN', 'RYR2', 'LRP1B', 'MUC16', 'ZFHX4', 'USH2A', 'CSMD3', 'NAV3', 'PCDH15', 'COL11A1', 'CSMD1', 'SYNE1', 'EYS', 'MUC17', 'ANKRD30B','FAM135B', 'FSIP2', 'TMEM132D'},
            'clinical_numerical_ids': ['overall_survival',  'vital_status','age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth'],
            'clinical_categorical_ids': ['gender', 'race', 'ethnicity']
        },
        graph_dataset=False,
        ppi_score_name='escore',
        ppi_score_threshold=0.0
    )

def test_prepare_data(data_module):
    data_module.prepare_data()
    assert isinstance(data_module.genomic_data, pd.DataFrame)
    assert isinstance(data_module.clinical_data, pd.DataFrame)
    assert isinstance(data_module.overall_survivals, pd.Series)
    assert isinstance(data_module.disease_specific_survivals, pd.Series)
    assert isinstance(data_module.primary_sites, pd.Series)

def test_get_patient_ids(data_module):
    data_module.prepare_data()
    data_module.get_patient_ids()
    print(type(data_module.patient_ids))
    assert isinstance(data_module.patient_ids, np.ndarray)
    #assert len(data_module.patient_ids) > 0

def test_get_clinical_ids(data_module):
    data_module.prepare_data()
    data_module.get_clinical_ids()
    assert isinstance(data_module.clinical_features, pd.Index)
    assert len(data_module.clinical_features) > 0

def test_get_genomic_ids(data_module):
    data_module.prepare_data()
    data_module.get_genomic_ids()
    assert isinstance(data_module.genomic_features, pd.Index)
    assert len(data_module.genomic_features) > 0

def test_normalize_clinical_data(data_module):
    data_module.prepare_data()
    data_module.normalize_clinical_data()
    assert isinstance(data_module.clinical_data, pd.DataFrame)

def test_setup(data_module):
    data_module.setup()
    #assert isinstance(data_module.train_data, pd.DataFrame)
    #assert isinstance(data_module.val_data, pd.DataFrame)
    assert isinstance(data_module.test_data, pd.DataFrame)

# def test_train_dataloader(data_module):
#     data_module.setup()
#     train_dataloader = data_module.train_dataloader()
#     assert isinstance(train_dataloader, DataLoader)

# def test_val_dataloader(data_module):
#     data_module.setup()
#     val_dataloader = data_module.val_dataloader()
#     assert isinstance(val_dataloader, DataLoader)

def test_test_dataloader(data_module):
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)
    print(isinstance(test_dataloader, DataLoader))


def main_test():
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
   
    
   
    
    #add the external data
    #external_testing_data = ExternalDataModule(**config['external_datasets'])

    external_testing_data = ExternalDataModule(
        project_id=['SCLC'],
        data_dir='Data/sclc_ucologne_2015',
        cache_directory='Cache/SCLC',
        batch_size=128,
        num_workers=4,
        chosen_features={
            'gene_ids': {'TP53', 'RB1', 'TTN', 'RYR2', 'LRP1B', 'MUC16', 'ZFHX4', 'USH2A', 'CSMD3', 'NAV3', 'PCDH15', 'COL11A1', 'CSMD1', 'SYNE1', 'EYS', 'MUC17', 'ANKRD30B','FAM135B', 'FSIP2', 'TMEM132D'},
            'clinical_numerical_ids': ['overall_survival',  'vital_status','age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth'],
            'clinical_categorical_ids': ['gender', 'race', 'ethnicity']
        },
        graph_dataset=False,
        ppi_score_name='escore',
        ppi_score_threshold=0.0
    )
    external_testing_data.setup()
     #project_id, data_dir, cache_directory, batch_size, num_workers, chosen_features=dict(),  
    # graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0
    test = external_testing_data.test_dataloader()
    
    # test iterating through the dataloader and check that it is not empty
    for i, batch in enumerate(test):
        print(i)
        print(batch)
        assert len(batch) > 0

if __name__ == '__main__':
    #pytest.main(['-sv', __file__])
    main_test()


    
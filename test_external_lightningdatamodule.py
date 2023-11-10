import pytest
import pandas as pd
from torch.utils.data import DataLoader
from external_lightningdatamodule import ExternalDataModule
import numpy as np

@pytest.fixture
def data_module():
    return ExternalDataModule(
        project_id=['SCLC'],
        data_dir='Data/sclc_ucologne_2015',
        cache_directory='Cache/SCLC',
        batch_size=32,
        num_workers=4,
        chosen_features={
            'gene_ids': {'ENSG00000141510', 'ENSG00000141510'},
            'clinical_numerical_ids': ['overall_survival',  'vital_status'],
            'clinical_categorical_ids': ['gender']
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
    assert isinstance(data_module.train_data, pd.DataFrame)
    assert isinstance(data_module.val_data, pd.DataFrame)
    assert isinstance(data_module.test_data, pd.DataFrame)

def test_train_dataloader(data_module):
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)

def test_val_dataloader(data_module):
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)

def test_test_dataloader(data_module):
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
    

    
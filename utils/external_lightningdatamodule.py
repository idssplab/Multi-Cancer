import pytorch_lightning as pl
from torch.utils import data
import pandas as pd
from utils.api import get_filters_result_from_project, get_ppi_encoder, get_network_image, visualize_ppi
from utils.logger import get_logger
from utils.util import check_cache_files
import numpy as np


class ExternalDataModule(pl.LightningDataModule):
    def __init__(self, project_id, data_dir, cache_directory, batch_size, num_workers, chosen_features=dict(),  graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0):
        #numworkers comes from cache directory
        super().__init__()
        self.data_dir = data_dir
        self.cache_directory = cache_directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_id = project_id
        self.target_type = 'overall_survival'
        self.n_threads = 1
        self.genomic_type = 'tpm'
        self.genomic_data = None
        self.clinical_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.logger = get_logger('preprocess.tcga_program_dataset')       
        

        
        self.get_chosen_features(chosen_features)
               

        # Specify the genomic type (use graph or not).
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold

        self.log_data_info()

        # Get data from TCGA_Project instance
        self._getdata()
 



    def get_chosen_features(self, chosen_features):
        # Get chosen features             
        self.chosen_project_gene_ids = chosen_features.get('gene_ids', {})
        self.chosen_clinical_numerical_ids: list = chosen_features.get('clinical_numerical_ids', [])
        self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

    def prepare_data(self):
        # Download the necessary data files
        # load sclc_ucologne_2015 data
        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm.tsv')
        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.tsv')
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch

    def log_data_info(self):
                # Log the information of the dataset.
        self.logger.info('Creating a TCGA Program Dataset with {} Projects...'.format(len(self.project_ids)))
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.patient_ids), len(self.genomic_ids), len(self.clinical_ids)
        ))
        self.logger.info('Target Type {}'.format(self.target_type)) #Target Type overall_survival
        self.logger.info('Overall survival imbalance ratio {} %'.format(
            sum(self.overall_survivals) / len(self.overall_survivals) * 100
        ))
        self.logger.info('Disease specific survival event rate {} %'.format(
            sum(self.disease_specific_survivals >= 0) / len(self.disease_specific_survivals) * 100
        ))
        self.logger.info('Disease specific survival imbalance ratio {} %'.format(
            sum(self.disease_specific_survivals[self.disease_specific_survivals >= 0]) / len(
                self.disease_specific_survivals[self.disease_specific_survivals >= 0]
            ) * 100
        ))
        self.logger.info('{} kinds of primary sites {}'.format(
            len(np.unique(self.primary_sites)), ' / '.join(self.primary_site_ids)
        ))
    

    def setup(self, stage=None):
        # Load the data files and split them into train, validation, and test sets
        #this dataset is only for testing 
        #self.test_data =  
        pass

    def DataLoader(self, data, shuffle=True):
        return data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prepare_batch=self.prepare_batch
        )

    def train_dataloader(self):
        return self.DataLoader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.DataLoader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self.DataLoader(self.test_data, shuffle=False)

    def collate_fn(self, batch):
        # Customize how the data is collated into batches
        pass

    def prepare_batch(self, batch):
        # Customize how the data is prepared for the model
        pass

    def teardown(self, stage=None):
        # Clean up any resources used by the data module
        pass
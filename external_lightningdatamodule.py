import pytorch_lightning as pl
from torch.utils import data
import pandas as pd
from utils.api import get_filters_result_from_project, get_ppi_encoder, get_network_image, visualize_ppi
from utils.logger import get_logger
from utils.util import check_cache_files
import numpy as np
from torch import from_numpy
import dgl


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
        self.patient_ids = None
        self.genomic_features = None
        self.clinical_features = None
        self.overall_survivals = None
        self.disease_specific_survivals = None
        self.primary_sites = None
        self.primary_site_ids = 0 # this will need to change


        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.logger = get_logger('preprocess.tcga_program_dataset')       
        

        
        self.get_chosen_features(chosen_features)
               

        # Specify the genomic type (use graph or not).
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold

        

       
        self.prepare_data()

        self.get_patient_ids()
        self.get_clinical_ids()
        self.get_genomic_ids()

        
        self.normalize_clinical_data()
        self.log_data_info()
        
 



    def get_chosen_features(self, chosen_features):
        # Get chosen features             
        self.chosen_project_gene_ids = chosen_features.get('gene_ids', {})
        self.chosen_clinical_numerical_ids: list = chosen_features.get('clinical_numerical_ids', [])
        self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

    def prepare_data(self):
        # Download the necessary data files
        # load sclc_ucologne_2015 data
        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm.tsv', header=0, sep='\t')
        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.tsv', header=0, sep='\t')
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        self.overall_survivals = self.clinical_data['overall_survival']
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival']
        self.primary_sites = self.clinical_data['primary_site']

        df_genomics = []
        df_clinicals = []
        df_vital_statuses = []
        df_overall_survivals = []
        df_disease_specific_survivals = []
        df_survival_times = []
        df_primary_sites = []
        df_project_ids = []
        train_patient_ids = []
        test_patient_ids = []
    
    def get_patient_ids(self):
        # Get the patient IDs
        self.patient_ids = np.unique(self.clinical_data['PATIENT_ID'])
        #print(self.patient_ids)
        self.logger.info('Total {} patients'.format(len(self.patient_ids)))

    def get_clinical_ids(self):
        self.clinical_features = self.clinical_data.columns[1:]
        #print(self.clinical_features)
    
    def get_genomic_ids(self):
        print(self.genomic_data.columns)
        self.genomic_features = self.genomic_data.columns

    def _process_genomic_as_graph(self, df_genomic: pd.DataFrame, df_ppi: pd.DataFrame):
        src = from_numpy(df_ppi['src'].to_numpy())
        dst = from_numpy(df_ppi['dst'].to_numpy())
        graphs: list[dgl.DGLGraph] = []

        # Create a graph for each sample (patient).
        for _, row in df_genomic.iterrows():
            g = dgl.graph((src, dst), num_nodes=self._num_nodes)
            g.ndata['feat'] = from_numpy(row.to_numpy()).view(-1, 1).float()
            g = dgl.add_reverse_edges(g)
            graphs.append(g)
        return graphs

    def normalize_clinical_data(self):
        self.logger.info('Normalize clinical numerical data using all samples')
        # Impute the missing values with mean
        
        clinical_mean = self.clinical_data[self.chosen_clinical_numerical_ids].mean()
        clinical_std = self.clinical_data[self.chosen_clinical_numerical_ids].std()
        # Impute the missing values with mean
        self.clinical_data = self.clinical_data.fillna(clinical_mean.to_dict())

        # Normalize the numerical values
        self.clinical_data[self.chosen_clinical_numerical_ids] -= clinical_mean
        self.clinical_data[self.chosen_clinical_numerical_ids] /= clinical_std

        self.clinical_data = pd.get_dummies(self.clinical_data, columns=self.chosen_clinical_categorical_ids, dtype=float)
        self.clinical_data = self.clinical_data.reindex(
                        columns=self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids
                    ).fillna(0)
        

    def log_data_info(self):
                # Log the information of the dataset.
        self.logger.info('Creating a TCGA Program Dataset with {} Projects...'.format(len(self.project_id)))
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.patient_ids), len(self.genomic_features), len(self.clinical_features)
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

# test that the module works

if __name__ == '__main__':
    data_module = ExternalDataModule(project_id='SCLC', data_dir='Data/sclc_ucologne_2015', cache_directory='cache', batch_size=32, num_workers=4, chosen_features=dict(),  graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0)
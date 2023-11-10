import pytorch_lightning as pl
from torch.utils import data
import pandas as pd
from utils.api import get_filters_result_from_project, get_ppi_encoder, get_network_image, visualize_ppi
from utils.logger import get_logger
from utils.util import check_cache_files
import numpy as np
from torch import from_numpy
import dgl
import torch
import shutil


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
        self.chosen_features = chosen_features

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
        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm.csv', header=1, sep=',')
       
        
        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.tsv', header=0, sep='\t')
        #PATIENT_ID	gender	ethnicity	race	year_of_diagnosis	year_of_birth	
        # overall_survival	vital_status	disease_specific_survival	primary_site
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        self.overall_survivals = self.clinical_data['overall_survival']
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival']
        self.primary_sites = self.clinical_data['primary_site']
        self


    
    def get_patient_ids(self):
        # Get the patient IDs
        self.patient_ids = np.unique(self.clinical_data['PATIENT_ID'])
        print(type(self.patient_ids))
        self.logger.info('Total {} patients'.format(len(self.patient_ids)))

    def get_clinical_ids(self):
        self.clinical_features = self.clinical_data.columns[1:]
        #print(self.clinical_features)
    
    def get_genomic_ids(self):
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
        self.prepare_data()
        self.get_chosen_features(self.chosen_features)
        self.log_data_info()
        self.normalize_clinical_data()
        self.split_data()

    def split_data(self, only_test = False):
            # Split the data into train, validation, and test sets
            if not only_test:
                self.logger.info('Splitting data into train, validation, and test sets...')
                train_data, val_data, test_data = [], [], []
                for project_id in self.project_id:
                    project_data = self.data[self.data['project_id'] == project_id]
                    project_data = project_data.sample(frac=1, random_state=self.random_state)
                    num_samples = len(project_data)
                    num_train_samples = int(num_samples * self.train_ratio)
                    num_val_samples = int(num_samples * self.val_ratio)
                    num_test_samples = num_samples - num_train_samples - num_val_samples
                    train_data.append(project_data.iloc[:num_train_samples])
                    val_data.append(project_data.iloc[num_train_samples:num_train_samples + num_val_samples])
                    test_data.append(project_data.iloc[num_train_samples + num_val_samples:])
                self.train_data = pd.concat(train_data)
                self.val_data = pd.concat(val_data)
                self.test_data = pd.concat(test_data)
            else:
                self.logger.info('Splitting data into test set...')
                test_data = []
                for project_id in self.project_id:
                    project_data = self.data[self.data['project_id'] == project_id]
                    project_data = project_data.sample(frac=1, random_state=self.random_state)
                    num_samples = len(project_data)
                    num_test_samples = num_samples
                    test_data.append(project_data.iloc[num_test_samples:])
                self.test_data = pd.concat(test_data)
        


    def DataLoader(self, data, shuffle=True):
        #DataLoader uses both the dataset and a sampler to determine the sequence of data loading.
        #The sampler is optional, but it allows you to specify the order indices are presented to your model.
        #The dataset is required and specifies the data elements that will be loaded.
        #The DataLoader class is used to wrap an existing Dataset during training and provides
        #additional utility methods to iterate through the data.

        # the data corresponds to the clinical and genomic data
        # the target corresponds to the vital_status

        

        return data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )



    def train_dataloader(self):
        return self.DataLoader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.DataLoader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self.DataLoader(self.test_data, shuffle=False)

    def collate_fn(self, batch):
        # Customize how the data is collated into batches
        # Unzip the data_list into two lists containing the two types of tuples
        graph_data_list, target_data_list = zip(*batch)
        # Unzip each list of tuples into separate lists

        graphs, clinicals, indices, project_ids = zip(*graph_data_list)
        targets, overall_survivals, vital_statuses = zip(*target_data_list)

        batched_graphs = batch(graphs)
        batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
        batch_indices = torch.tensor(indices)
        batch_project_ids = torch.tensor(project_ids)
        batch_targets = torch.tensor(targets)
        batch_overall_survivals = torch.tensor(overall_survivals)
        batch_vital_statuses = torch.tensor(vital_statuses)
        return ((batched_graphs, batch_clinicals, batch_indices, batch_project_ids),
                (batch_targets, batch_overall_survivals, batch_vital_statuses))
        

    def prepare_batch(self, batch):
        # Customize how the data is prepared for the model
        #The batches have this shape
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch

        # Unpack the batch
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        
        # Convert the data to PyTorch tensors
        genomic = torch.from_numpy(genomic).float()
        clinical = torch.from_numpy(clinical).float()
        index = torch.tensor(index).long()
        project_id = torch.tensor(project_id).long()
        overall_survival = torch.tensor(overall_survival).float()
        survival_time = torch.tensor(survival_time).float()
        vital_status = torch.tensor(vital_status).float()
        
        return (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status)

    def teardown(self, stage=None):
        # Clean up any resources used by the data module
        if stage == 'fit' or stage is None:
            shutil.rmtree(self.cache_directory)

# test that the module works

if __name__ == '__main__':
    data_module = ExternalDataModule(project_id='SCLC', data_dir='Data/sclc_ucologne_2015', cache_directory='cache', batch_size=32, num_workers=4, chosen_features=dict(),  graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0)
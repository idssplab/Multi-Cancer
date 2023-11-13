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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate


# Create a TensorDataset from the tensors
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, genomic_features, clinical_features):
        self.data = data
        self.genomic_features = genomic_features
        self.clinical_features = clinical_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
            # Assuming self.data is a pandas DataFrame
            row = self.data.iloc[index]
            genomic = row[self.genomic_features]
            clinical = row[self.clinical_features]
            index = index#row['PATIENT_ID']
            project_id = row['project_id']
            overall_survival = row['overall_survival']
            survival_time = row['survival_time']
            vital_status = row['vital_status']
            
            return (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status)





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
        self.chosen_clinical_numerical_ids= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.chosen_clinical_categorical_ids = ['gender' ,'race', 'ethnicity']
        self.all_clinical_feature_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

 

        self.data = None
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
        self.vital_status = None

        self._genomics = None
        self._clinicals = None
        self.targets =  None
        self._survival_times =  None
        self._vital_statuses =  None


        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.logger = get_logger('preprocess.tcga_program_dataset')     

       
        self.pin_memory = True

        
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
        # self.chosen_clinical_numerical_ids: list = chosen_features.get('clinical_numerical_ids', [])
        # self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])

        self.chosen_clinical_numerical_ids= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.chosen_clinical_categorical_ids = ['gender' ,'race', 'ethnicity']
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

    def prepare_data(self):
        # Download the necessary data files
        # load sclc_ucologne_2015 data
        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm.csv', header=1, sep=',')
       
        
        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.csv', header=0, sep=',')
        #PATIENT_ID	gender	ethnicity	race	year_of_diagnosis	year_of_birth	
        # overall_survival	vital_status	disease_specific_survival	primary_site
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        #print(self.clinical_data.columns)
        self.overall_survivals = self.clinical_data.overall_survival
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival']
        self.primary_sites = self.clinical_data['primary_site']
        self.vital_status = self.clinical_data['vital_status']
        

        


    def __getitem__(self, index):
        return (self._genomics[index], self._clinicals[index], index, 0), (self.targets[index], self._survival_times[index], self._vital_statuses[index])


    
    def get_patient_ids(self):
        # Get the patient IDs
        self.patient_ids = np.unique(self.clinical_data['PATIENT_ID'])
       
        self.logger.info('Total {} patients'.format(len(self.patient_ids)))

    def get_clinical_ids(self):
        self.clinical_features = self.clinical_data.columns[1:]
        #print(self.clinical_features)
    
    def get_genomic_ids(self):
        self.genomic_features = self.genomic_data.columns[1:]

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
        #['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        
        clinical_mean = self.clinical_data[self.chosen_clinical_numerical_ids].mean()
        clinical_std = self.clinical_data[self.chosen_clinical_numerical_ids].std()
        # Impute the missing values with mean
        self.clinical_data = self.clinical_data.fillna(clinical_mean.to_dict())

        # Normalize the numerical values
        self.clinical_data[self.chosen_clinical_numerical_ids] -= clinical_mean
        self.clinical_data[self.chosen_clinical_numerical_ids] /= clinical_std

        self.clinical_data = pd.get_dummies(self.clinical_data, columns=self.chosen_clinical_categorical_ids, dtype=float)
        print('clinical data', self.clinical_data.columns)

        self.clinical_data = self.clinical_data.select_dtypes(exclude=['object'])
        self.all_clinical_feature_ids = self.clinical_data.columns

        

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

    def concat_data(self):
        # Concatenate the genomic and clinical data , having the genes and clinical features as columns
        
        
        self.data = pd.merge(self.clinical_data, self.genomic_data , left_index=True, right_index=True)

        self.data['overall_survival'] = self.overall_survivals
        self.data['survival_time'] = self.disease_specific_survivals
        self.data['vital_status'] = self.vital_status
        self.data['project_id'] = self.project_id

        #get rid of object type columns
        self.data = self.data.select_dtypes(exclude=['object'])       
        
        self.logger.info('Total {} samples'.format(len(self.data)))

    

    def setup(self, stage=None):
        # Load the data files and split them into train, validation, and test sets
        #this dataset is only for testing 
        self.prepare_data()
        self.get_chosen_features(self.chosen_features)
        self.log_data_info()
        self.normalize_clinical_data()
        self.concat_data()
        self.split_data()
        self.create_tensors()

    def create_tensors(self):
        # if any column is object type, print names of columns with object type
        dtypes = self.genomic_data.dtypes
        object_cols = dtypes[dtypes == 'object'].index
        print(object_cols)
        #get rid of the ID column
        self.genomic_data = self.genomic_data.drop(columns=['gene_id'])

        self._genomics = torch.tensor(self.genomic_data.values, dtype=torch.float32)
        self._clinicals = torch.tensor(self.clinical_data.values, dtype=torch.float32)
        self.targets = torch.tensor(self.overall_survivals.values, dtype=torch.float32)
        self._survival_times = torch.tensor(self.disease_specific_survivals.values, dtype=torch.float32)
        self._vital_statuses = torch.tensor(self.vital_status.values, dtype=torch.float32)

    def split_data(self, only_test = True):
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
                # (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
                self.test_data = self.data
        


    def DataLoader(self, data, shuffle=True):
        

        data = self.test_data

        # print(self.test_data.dtypes)
        # # # Check which columns are object type
        # object_cols = data.select_dtypes(include=['object']).columns

        # # Select the columns that are object type
        # df_object = data[object_cols]

        

        # df_object = df_object.astype('category')

        # #incorporate the categorical columns into the dataframe
        # data[object_cols] = df_object

        # # Use one hot encoding to convert the categorical columns to numerical
        
        # data = pd.get_dummies(data, columns=object_cols, dtype=float)
        
        # Convert them to tensor
        #tensor_object = torch.tensor([df_object[col].cat.codes.values for col in df_object], dtype=torch.int64).T

        #features = torch.tensor(data.values)
        #features = torch.tensor(data[self.clinical_features + self.genomic_features].values, dtype=torch.float32)
        dataset = CustomDataset(data=data, genomic_features=self.genomic_features, clinical_features=self.all_clinical_feature_ids)
        
        #targets = torch.tensor(data[self.overall_survivals].values, dtype=torch.float32)

        # Create a DataLoader from the TensorDataset
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.prepare_batch,
            pin_memory=True,
        )


        return dataloader


    def train_dataloader(self):
        return self.DataLoader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.DataLoader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self.DataLoader(self.test_data, shuffle=False, )

    # def collate_fn(self, batch, graph_dataset=False):
    #     # Customize how the data is collated into batches
    #     # Unzip the data_list into two lists containing the two types of tuples

    #     if graph_dataset:
    #         graph_data_list, target_data_list = zip(*batch)     
    #         graphs, clinicals, indices, project_ids = zip(*graph_data_list)
    #         targets, overall_survivals, vital_statuses = zip(*target_data_list)

    #         batched_graphs = batch(graphs)
    #         batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
    #         batch_indices = torch.tensor(indices)
    #         batch_project_ids = torch.tensor(project_ids)
    #         batch_targets = torch.tensor(targets)
    #         batch_overall_survivals = torch.tensor(overall_survivals)
    #         batch_vital_statuses = torch.tensor(vital_statuses)
    #         return ((batched_graphs, batch_clinicals, batch_indices, batch_project_ids),
    #                 (batch_targets, batch_overall_survivals, batch_vital_statuses))
    #     else:
    #         print('Using non graph dataset collate function')
    #         # gene_data_list, target_data_list = zip(*batch)  
    #         # genes, clinicals, indices, project_ids = zip(*gene_data_list)
    #         # targets, overall_survivals, vital_statuses = zip(*target_data_list)


    #         # batched_genes = torch.tensor(genes)
    #         # batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
    #         # batch_indices = torch.tensor(indices)
    #         # batch_project_ids = torch.tensor(project_ids)
    #         # batch_targets = torch.tensor(targets)
    #         # batch_overall_survivals = torch.tensor(overall_survivals)
    #         # batch_vital_statuses = torch.tensor(vital_statuses)
    #         # return ((batched_genes, batch_clinicals, batch_indices, batch_project_ids),
    #         #         (batch_targets, batch_overall_survivals, batch_vital_statuses))
    #         return default_collate(batch)
        
    def collate_fn(self, batch, graph_dataset=False):
        # Customize how the data is collated into batches
        # Unzip the data_list into two lists containing the two types of tuples

        if graph_dataset:
            graph_data_list, target_data_list = zip(*batch)
            graphs, clinicals, indices, project_ids = zip(*graph_data_list)
            targets, overall_survivals, vital_statuses = zip(*target_data_list)

            batched_graphs = batch(graphs)
            batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
            batch_indices = torch.tensor(indices)
            batch_project_ids = torch.tensor(project_ids)
        else:
            data_list = list(batch)
            features = torch.stack([torch.from_numpy(data[self.clinical_features + self.genomic_features].values) for data in data_list])
            targets = torch.stack([torch.from_numpy(data[self.overall_survivals].values) for data in data_list])
            batch = (features, targets)

        return batch
        


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



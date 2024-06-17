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
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, WeightedRandomSampler, Subset
from torch.utils.data.dataloader import default_collate
import warnings
from sklearn.model_selection import train_test_split
from inmoose.pycombat import pycombat_seq 
from sklearn.model_selection import KFold
from datasets_manager.sampler import BootstrapSubsetSamplerDM, create_stratified_sampler, compute_weights_by_task_id
from dgl import batch
from torch.utils.data import Subset, DataLoader

warnings.filterwarnings("ignore", ".*sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.*")

SEED = 1126 # sklearn should use the np seed


def gcollate(data_list):
    # Unzip the data_list into two lists containing the two types of tuples
    graph_data_list, target_data_list = zip(*data_list)
    # Unzip each list of tuples into separate lists
    graphs, clinicals, indices, project_ids = zip(*graph_data_list)
    targets, survival_times, vital_statuses = zip(*target_data_list)

    batched_graphs = batch(graphs)
    batch_clinicals = torch.stack([torch.from_numpy(clinical) for clinical in clinicals])
    batch_indices = torch.tensor(indices)
    batch_project_ids = torch.tensor(project_ids)
    batch_targets = torch.tensor(targets)
    batch_survival_times = torch.tensor(survival_times)
    batch_vital_statuses = torch.tensor(vital_statuses)
    return ((batched_graphs, batch_clinicals, batch_indices, batch_project_ids),
            (batch_targets, batch_survival_times, batch_vital_statuses))

def normalize_dataset_combat(pre_normalized_rna):
    num_samples = pre_normalized_rna.shape[0]
       
    # TODO: merge Ian's code for batch correction
    # only works with more than 1 batch
    # temporary fix: associating half of the samples to one batch and the other half to the other batch
    batch1_n = int(np.ceil(num_samples * 0.5))
    batch2_n = int(num_samples - batch1_n)   
    batches = ["Batch 1"] * batch1_n  + ["Batch 2"] * batch2_n
    # shuffle the batches numbers
    batches = np.random.permutation(batches)
    
    # Apply ComBat
    normalized_rna = pycombat_seq(pre_normalized_rna.T, batches)
    normalized_rna = normalized_rna.T
    return normalized_rna


def check_for_categorical_zeros(df):
    race_cols = ["race_not reported", "race_white", "race_asian", "race_american indian or alaska native", "race_black or african american", "race_native hawaiian or other pacific islander"]           
    ethnicity_cols = ["ethnicity_not reported", "ethnicity_not hispanic or latino", "ethnicity_hispanic or latino"]

    # if all values in a row are zero, then the sum of the row will be zero
    if sum(df[race_cols].all(axis=1)) == 0:
        # grab all rows where the sum of the row is zero
        # for these rows, set the value of "race_not reported" to 1
        df.loc[df[race_cols].sum(axis=1) == 0, "race_not reported"] = 1
        
    if sum(df[ethnicity_cols].all(axis=1)) == 0:
        df.loc[df[ethnicity_cols].sum(axis=1) == 0, "ethnicity_not reported"] = 1
        df.loc[df[ethnicity_cols].sum(axis=1) == 0, "ethnicity_not reported"] = 1
        
    return df

# Create a TensorDataset from the tensors
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, genomic_features, clinical_features, graph_dataset =True, ppi_score_name='escore', ppi_score_threshold=0.0):
        self.data = data
        self.genomic_features = genomic_features
        self.clinical_features = clinical_features
        self.genomic_data = data[self.genomic_features]
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold

        if graph_dataset:
            self._num_nodes = self.genomic_data.shape[-1]
            print(f'Number of nodes for the graph: {self._num_nodes}')
            df_ppis = get_ppi_encoder(self.genomic_data.columns.to_list(), score=self.ppi_score, threshold=self.ppi_threshold)
            #get_network_image(df_genomics.columns.to_list())
            
            self.genomic_data = self._process_genomic_as_graph(self.genomic_data, df_ppis)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
            
            # Assuming self.data is a pandas DataFrame
            row = self.data.iloc[index]
            if self.graph_dataset:
                
                genomic= self.genomic_data[index]
                
            else:
                genomic = row[self.genomic_features].values #sending ndarray            
            clinical = row[self.clinical_features].values           
            index = index#row['PATIENT_ID']
            project_id = row['project_id']
            overall_survival = row['overall_survival']
            survival_time = row['survival_time']
            vital_status = row['vital_status']
            
            return ((genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status))
    
    def get_index(self):
        return self.data.index.values
   
    def get_targets(self):
        overall_survival = self.data['overall_survival'].values
        return overall_survival
    
    def get_features(self):
        data_without_targets = self.data.drop(columns=['overall_survival', 'survival_time', 'vital_status'])
        return data_without_targets
    
    def get_project_ids(self):
        return self.data['project_id'].values
    
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





class DataModule(pl.LightningDataModule):
    def __init__(self, project_ids, data_dir, cache_directory, batch_size, num_workers, chosen_features=dict(),  
                 graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0, project_id_task_descriptor=0, test_split=0.2, 
                 n_threads=16, no_validation_set = False, multi_task = True, use_repeated_genes = True, batch_correction = False, os_threshold = 60):    
        #numworkers comes from cache directory
        super().__init__()
        self.project_id_task_descriptor = project_id_task_descriptor
        self.data_dir = data_dir
        self.cache_directory = cache_directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_ids = project_ids
        self.target_type = 'overall_survival'
        self.n_threads = n_threads
        self.chosen_features = chosen_features
        self.chosen_genes = chosen_features['gene_ids']
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold
        self.collate_fn = default_collate
        self.batch_correction = batch_correction
        self.os_threshold = os_threshold

        if self.graph_dataset:
            self.collate_fn = gcollate
        

        self.clinical_numerical_features= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.clinical_categorical_features = ['gender' ,'race', 'ethnicity']
        self.clinical_features = self.clinical_numerical_features + self.clinical_categorical_features
        self.test_split = test_split
        self.no_validation_set = no_validation_set
        self.use_old_TCGA_data = True # the old data comes with the already transformed to binary overall survival and disease specific survival
        self.number_of_sets = len(self.project_ids)
        self.use_repeated_genes = False # used for the old multi-task dataset, non graph


        if multi_task:
            # create a dictionary of project ids and task descriptors, to each element of the list of project ids, assign a task descriptor
            # this is for the multi-task dataset
            self.project_id_task_descriptor = dict()
            for i, cancer_id in enumerate(self.project_ids):
                self.project_id_task_descriptor[cancer_id] = i

        
                
  
        self.all_genomic_features = []
        self.data = dict() # for multitask, this will be a dictionary of dataframes
        self.genomic_type = 'tpm'
        self.genomic_data = dict()# for multitask, this will be a dictionary of dataframes
        self.clinical_data = dict()# for multitask, this will be a dictionary of dataframes
        self.patient_ids = dict()
        self.genomic_features = dict()
        self.overall_survivals = dict()
        self.disease_specific_survivals = dict()
        self.primary_sites = dict()
        self.primary_site_ids = 0 # this will need to change
        self.vital_status = dict()

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
               

        # Specify the genomic type (use graph or not).
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold        

        #self.get_chosen_features()
        self.prepare_data()        
        self.get_patient_ids()        
        self.get_genomic_ids()     
        


    def get_chosen_features(self):
        # Get chosen features    
        chosen_features = self.chosen_features         
        self.chosen_genes =  chosen_features['gene_ids']    
        self.clinical_numerical_features= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.clinical_categorical_features = ['gender' ,'race', 'ethnicity']
        self.chosen_clinical_ids = self.clinical_numerical_features + self.clinical_categorical_features

    def prepare_data(self):
        
        
        for cancer_id in self.project_ids:
            
            self.genomic_data[cancer_id] = pd.read_csv(self.data_dir + cancer_id+ '/rna_data.csv', header=0, sep=',') # depends on the cancer directory       
            self.filter_genes(cancer_id) # keep only the 20 genes we are actually using
            self.clinical_data[cancer_id] = pd.read_csv(self.data_dir +cancer_id+ '/clin_data.csv', header=0, sep=',')
            self.primary_sites[cancer_id] = self.clinical_data[cancer_id]['primary_site']
        
                
    def filter_genes(self, cancer_id):      
        if 'Unnamed: 0' not in self.genomic_data[cancer_id].columns:
            self.genomic_data[cancer_id] = self.genomic_data[cancer_id].reset_index()
        else:
            self.genomic_data[cancer_id] = self.genomic_data[cancer_id][['Unnamed: 0']+ self.chosen_genes[cancer_id] ] 
            self.genomic_data[cancer_id] = self.genomic_data[cancer_id].drop(columns=['Unnamed: 0'])
        
        # sort the genes alphabetically
        self.genomic_data[cancer_id] = self.genomic_data[cancer_id].sort_index(axis=1)
  
    
    def get_patient_ids(self):
        # Get the patient IDs
        for cancer_id in self.project_ids:
            self.patient_ids[cancer_id] = np.unique(self.clinical_data[cancer_id]['PATIENT_ID'])       
            self.logger.info('Total {} patients'.format(len(self.patient_ids[cancer_id])))

    
    def get_genomic_ids(self):
        
        for cancer_id in self.project_ids:
            self.genomic_features[cancer_id] = self.genomic_data[cancer_id].columns
            self.all_genomic_features.extend(self.genomic_features[cancer_id])
        
        if self.use_repeated_genes:
            # keep the repeated genes
            self.all_genomic_features = self.all_genomic_features
            print('Number of genomic features', len(self.all_genomic_features))
        else:
            self.all_genomic_features = list(set(self.all_genomic_features)) # remove duplicates
        

        #self.logger.info('Total {} genomic features'.format(len(self.genomic_features)))



    def remove_nan_values(self):


        for cancer_id in self.project_ids:
        #remove nan values from vital status, replace nan values with 0
            self.clinical_data[cancer_id]['vital_status'] = self.clinical_data[cancer_id]['vital_status'].fillna(0)
            self.clinical_data[cancer_id]['survival_time'] = self.clinical_data[cancer_id]['survival_time'].fillna(0)

        
      
    def transform_survival_data(self):

        self.remove_nan_values()
        # Transform the disease specific survival and overall survival to binary
        if not self.use_old_TCGA_data:            
            months_threshold = 60 # 5 years 
            for cancer_id in self.project_ids:
                self.clinical_data[cancer_id]['disease_specific_survival'] = (self.clinical_data[cancer_id]['disease_specific_survival'] < months_threshold).astype(int)        
                self.clinical_data[cancer_id]['overall_survival'] = (self.clinical_data[cancer_id]['overall_survival'] < months_threshold).astype(int) #target
        if self.os_threshold != 60:
            for cancer_id in self.project_ids:
                # save a csv with the old and new values, and the survival time and vital status
                df = pd.DataFrame({'old_overall_survival': self.clinical_data[cancer_id]['overall_survival'], 'new_overall_survival': (self.clinical_data[cancer_id]['survival_time'] < self.os_threshold).astype(int), 'survival_time': self.clinical_data[cancer_id]['survival_time'], 'vital_status': self.clinical_data[cancer_id]['vital_status']})
                print('Old imbalance ratio for cancer {} is {}'.format(cancer_id, self.clinical_data[cancer_id]['overall_survival'].value_counts(normalize=True)[1]*100))
                self.clinical_data[cancer_id]['overall_survival'] = (self.clinical_data[cancer_id]['survival_time'] < self.os_threshold).astype(int)
                print('New imbalance ratio for cancer {} is {}'.format(cancer_id, self.clinical_data[cancer_id]['overall_survival'].value_counts(normalize=True)[1]*100))
                df['new_overall_survival'] = self.clinical_data[cancer_id]['overall_survival']  
              
                df.to_csv('check_survival_time.csv', index=False)

             


        # Keep as separate columns for the tensors
        for cancer_id in self.project_ids:
            self.overall_survivals[cancer_id] = self.clinical_data[cancer_id]['overall_survival'] 
            self.disease_specific_survivals[cancer_id] = self.clinical_data[cancer_id]['disease_specific_survival'] 
            self.vital_status[cancer_id] = self.clinical_data[cancer_id]['vital_status']  

    def process_categorical_clinical_data(self):
        #assigned directly so that the order is preserved
        clin_col_names = ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth', 
            'gender_female', 'gender_male', 'race_american indian or alaska native', 'race_asian', 'race_black or african american',
            'race_not reported', 'race_white', 'ethnicity_hispanic or latino', 
            'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'race_native hawaiian or other pacific islander']

        total_clin_col_names = ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth',
        'overall_survival', 'vital_status', 'disease_specific_survival',
        'survival_time', 'gender_female', 'gender_male', 'race_not reported',
        'race_white', 'race_asian', 'ethnicity_not reported',
        'ethnicity_not hispanic or latino',
        'race_american indian or alaska native',
        'race_black or african american', 'ethnicity_hispanic or latino',
        'race_native hawaiian or other pacific islander'] #this is the order of the columns in the csv file

        self.clinical_features =  clin_col_names
        #applied before concat        
        # CATEGORICAL COLS #['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth'] 
        for cancer_id in self.project_ids:     
            self.clinical_data[cancer_id] = pd.get_dummies(self.clinical_data[cancer_id], columns=self.clinical_categorical_features, dtype=float)             
            #change the column names to lower case
            self.clinical_data[cancer_id].columns = map(str.lower, self.clinical_data[cancer_id].columns)          
            self.clinical_data[cancer_id] = self.clinical_data[cancer_id].select_dtypes(exclude=['object'])   
            
            #if any of the columns is missing, add it with 0 values
            for col in clin_col_names:
                if col not in self.clinical_data[cancer_id].columns:
                    #print('adding column {}'.format(col))
                    self.clinical_data[cancer_id][col] = 0

            self.clinical_data[cancer_id] = check_for_categorical_zeros(self.clinical_data[cancer_id])     
            #reorder the columns           
            self.clinical_data[cancer_id] = self.clinical_data[cancer_id][total_clin_col_names]       
            

    def concat_data(self):
        # Concatenate the genomic and clinical data , having the genes and clinical features as columns       
        
        self.logger.info('Concatenating genomic and clinical data...')
        for cancer_id in self.project_ids:
            
            self.data[cancer_id] = pd.merge(self.clinical_data[cancer_id], self.genomic_data[cancer_id] , left_index=True, right_index=True)
            self.data[cancer_id]['project_id'] =self.project_id_task_descriptor[cancer_id]         
        
            #get rid of object type columns
            self.data[cancer_id] = self.data[cancer_id].select_dtypes(exclude=['object'])    
            self.log_data_info(cancer_id)
        
       
            
            
           


    def split_into_train_val_test(self, data):
        
        if self.no_validation_set:
            if self.test_split == 0:
                return data, data
            train, test = train_test_split(data, test_size=self.test_split, stratify=data['overall_survival'])            
            return train, test
        else:
            train, test = train_test_split(data, test_size=self.test_split, stratify=data['overall_survival'])
            train, val = train_test_split(train, test_size=self.test_split, stratify=train['overall_survival'])
            return train, val, test

    def split_data(self, only_test = False):
            # Split the data into train, validation, and test sets
            if not only_test: #this is for the MTL dataset mostly
                
                train_data, val_data, test_data = [], [], []
               
                
                #project id task descriptor {'BRCA': 0, 'LUAD': 1, 'COAD': 2}
                for cancer_id in self.project_ids: 
                                     
                    project_data = self.data[cancer_id]
                    #first row shuffle
                    project_data = project_data.sample(frac=1)
                   
                    # The selection preserves the proportion of original classes
                    if self.no_validation_set:
                        self.logger.info('Splitting data into train and test sets...')
                        train, test = self.split_into_train_val_test(project_data)
                        train_data.append(train)
                        test_data.append(test)
                    else:
                        self.logger.info('Splitting data into train, validation, and test sets...')
                        train, val, test = self.split_into_train_val_test(project_data)
                        val_data.append(val)
                        train_data.append(train)                    
                        test_data.append(test)

                # this is the concatenated dataframe, not Dataset or DataLoader
                self.train_data = pd.concat(train_data).fillna(0)  # the genes that don't exist in a cancer are filled with 0             
                self.test_data = pd.concat(test_data).fillna(0)
                if not self.no_validation_set:
                    self.val_data = pd.concat(val_data).fillna(0)
            else:
                self.logger.info('Splitting data into test set...')
                # (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
                self.test_data = self.data


    def preprocess_clinical_numeric_data(self):
        # we should normalize the clinical data for all cancers at the same time
        # Normalize the numerical values, using the training set
        clinical_data = self.train_data[self.clinical_numerical_features]      

        clinical_mean = clinical_data[self.clinical_numerical_features].mean()
        clinical_std = clinical_data[self.clinical_numerical_features].std()
        clinical_std = clinical_std.replace(0, 1e-6) # avoid dividing by zero

        # Impute the missing values with mean
        # for training, validation and test sets separately to avoid leakage
        self.train_data[self.clinical_numerical_features] = self.train_data[self.clinical_numerical_features].fillna(clinical_mean.to_dict())    
        self.train_data[self.clinical_numerical_features] = (self.train_data[self.clinical_numerical_features] - clinical_mean) / clinical_std
        
        self.test_data[self.clinical_numerical_features] = self.test_data[self.clinical_numerical_features].fillna(clinical_mean.to_dict())    
        self.test_data[self.clinical_numerical_features] = (self.test_data[self.clinical_numerical_features] - clinical_mean) / clinical_std
        if not self.no_validation_set:
            self.val_data[self.clinical_numerical_features] = self.val_data[self.clinical_numerical_features].fillna(clinical_mean.to_dict())    
            self.val_data[self.clinical_numerical_features] = (self.val_data[self.clinical_numerical_features] - clinical_mean) / clinical_std

    def normalize_genomic_data(self):
        # be sure to divide into train, val, test sets before normalizing            
        # batch correction for all sets separately
        # print("only batch correction is being applied to the genomic data")
        if self.data_dir != 'Data/cbio-tcga-COADREAD-2018/': # this DS comes batch normalized
                       
            # save csv before and after batch correction
            
            self.train_data[self.all_genomic_features] = normalize_dataset_combat(self.train_data[self.all_genomic_features].values)
            self.test_data[self.all_genomic_features] = normalize_dataset_combat(self.test_data[self.all_genomic_features].values)
            if not self.no_validation_set:
                self.val_data[self.all_genomic_features] = normalize_dataset_combat(self.val_data[self.all_genomic_features].values)
            
   
    def log_data_info(self, cancer_id):
        

        self.logger.info('{} - Total {} samples'.format(str(cancer_id),len(self.data[cancer_id])))
        self.logger.info('{} - Total {} features'.format(str(cancer_id),len(self.data[cancer_id].columns)))
        self.logger.info('{} - Total {} genomic features'.format(str(cancer_id),len(self.genomic_data[cancer_id].columns)))
        self.logger.info('{} - Total {} clinical features'.format(str(cancer_id),len(self.clinical_data[cancer_id].columns)))
        self.logger.info('{} - Overall survival imbalance ratio {} %'.format(str(cancer_id),
            sum(self.data[cancer_id]['overall_survival']) / len(self.data[cancer_id]['overall_survival']) * 100
        ))
        #check if there are any missing values
        self.logger.info('{} - Total {} missing values'.format(str(cancer_id),self.data[cancer_id].isnull().sum().sum()))
    
    def setup(self, stage=None):
        self.prepare_data()
        self.transform_survival_data()
        self.process_categorical_clinical_data() # one hot encode categories and change column names
        self.concat_data() # moved earlier, before the split in train and test
        self.split_data()
        self.get_chosen_features()    
        self.preprocess_clinical_numeric_data()
        if self.batch_correction:
            self.normalize_genomic_data()

    def create_sampler(self, weights, num_samples, replacement=True):
        weights = torch.DoubleTensor(weights)
        return WeightedRandomSampler(weights, num_samples, replacement)

        

    def DataLoader(self, data, weighted_sampler = False, replacement = False):
        shuffle=False
          
        dataset = CustomDataset(data=data, genomic_features=self.all_genomic_features, clinical_features=self.clinical_features, graph_dataset=self.graph_dataset, ppi_score_name=self.ppi_score, ppi_score_threshold=self.ppi_threshold)   
          
        
        if weighted_sampler: #using the targets
            weights = data['overall_survival'].values
            sampler = self.create_sampler(weights, len(dataset), replacement=True)
            #sampler = WeightedRandomSampler(data['overall_survival'].values, len(data), replacement=True)
            #self.logger.info('Using weighted sampler (by targets)')
        else:  
            # originally uses BootstrapSubsetSampler 
            #sampler = BootstrapSubsetSamplerDM(dataset, replacement=replacement, num_samples=len(dataset))  
            sampler = RandomSampler( data_source=dataset, replacement=True, num_samples=len(dataset))   
            #self.logger.info('Using random sampler')
       

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            sampler=sampler,

            
        )
        return dataloader
    


    
   
    def train_dataloader(self):

        # originally uses SubsetWeightedRandomSampler
        return self.DataLoader(self.train_data,  weighted_sampler=True, replacement=True) 
    

    def val_dataloader(self):
        #not being used currently
        return self.DataLoader(self.val_data, weighted_sampler=False, replacement=False)

    def test_dataloader(self):
        # originally uses BootstrapSubsetSampler
        return self.DataLoader(self.test_data, weighted_sampler=False, replacement=True)
    




    def teardown(self, stage=None):
        # Clean up any resources used by the data module
        if stage == 'fit' or stage is None:
            shutil.rmtree(self.cache_directory)



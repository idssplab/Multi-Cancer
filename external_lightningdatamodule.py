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
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, WeightedRandomSampler
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
            genomic = row[self.genomic_features].values #sending ndarray
            
            clinical = row[self.clinical_features].values
            
            index = index
            project_id = row['project_id']
            overall_survival = row['overall_survival']
            survival_time = row['survival_time']
            vital_status = row['vital_status']
            
            return ((genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status))





class ExternalDataModule(pl.LightningDataModule):
    def __init__(self, project_id, data_dir, cache_directory, batch_size, num_workers, chosen_features=dict(),  graph_dataset= False, ppi_score_name='escore', ppi_score_threshold=0.0, project_id_task_descriptor=0):
        #numworkers comes from cache directory
        super().__init__()
        self.project_id_task_descriptor = project_id_task_descriptor
        self.data_dir = data_dir
        self.cache_directory = cache_directory
        self.batch_size = batch_size
        
        self.num_workers = num_workers
        self.project_id = project_id
        self.target_type = 'overall_survival'
        self.n_threads = 1
        self.chosen_features = chosen_features
        self.chosen_genes = chosen_features['gene_ids']
        
        self.chosen_clinical_numerical_ids= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.chosen_clinical_categorical_ids = ['gender' ,'race', 'ethnicity']
        self.all_clinical_feature_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

 
        self.os_threshold = 60
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
        
               

        # Specify the genomic type (use graph or not).
        self.graph_dataset = graph_dataset
        self.ppi_score = ppi_score_name
        self.ppi_threshold = ppi_score_threshold        

        self.get_chosen_features(chosen_features)
        self.prepare_data()        
        self.get_patient_ids()
        self.get_clinical_ids()
        self.get_genomic_ids()        
        self.normalize_clinical_data()
        self.log_data_info()
        
 



    def get_chosen_features(self, chosen_features):
        # Get chosen features             
        self.chosen_genes =  list(chosen_features['gene_ids'])     
        self.chosen_clinical_numerical_ids= ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.chosen_clinical_categorical_ids = ['gender' ,'race', 'ethnicity']
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

    def prepare_data(self):

        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm_small.csv', header=0, sep=',')
       
        self.filter_genes()

        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.csv', header=0, sep=',')

        self.overall_survivals = self.clinical_data.overall_survival
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival']
        self.primary_sites = self.clinical_data['primary_site']
        self.vital_status = self.clinical_data['vital_status']
        

        
    def filter_genes(self):
        self.genomic_data = self.genomic_data[['Unnamed: 0']+ self.chosen_genes ]


   
    
    def get_patient_ids(self):
        # Get the patient IDs
        self.patient_ids = np.unique(self.clinical_data['PATIENT_ID'])
       
        self.logger.info('External DS - Total {} patients'.format(len(self.patient_ids)))

    def get_clinical_ids(self):
        self.clinical_features = self.clinical_data.columns[1:]
        #print(self.clinical_features)
    
    def get_genomic_ids(self):
        self.genomic_features = self.genomic_data.columns[1:]



    def preprocess_clinical_numeric_data(self):
        clinical_mean = self.clinical_data[self.chosen_clinical_numerical_ids].mean()
        clinical_std = self.clinical_data[self.chosen_clinical_numerical_ids].std()
        # Impute the missing values with mean
        self.clinical_data = self.clinical_data.fillna(clinical_mean.to_dict())


        clinical_std = clinical_std.replace(0, 1e-6)

        # Normalize the numerical values
        self.clinical_data[self.chosen_clinical_numerical_ids] -= clinical_mean
        #the std is 0 for year_of_diagnosis
        self.clinical_data[self.chosen_clinical_numerical_ids] /= clinical_std

        
        self.transform_survival_data()
                
        self.overall_survivals = self.clinical_data['overall_survival'] 
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival'] 
        self.vital_status = self.clinical_data['vital_status']


    def remove_nan_values(self):
        # Remove the rows with missing values in the clinical data
        self.clinical_data = self.clinical_data.dropna(subset=['overall_survival', 'disease_specific_survival', 'vital_status'])
        self.clinical_data = self.clinical_data.reset_index(drop=True)
        self.logger.info('External DS - Total {} samples after removing missing values'.format(len(self.clinical_data)))

    def transform_survival_data(self):

        self.remove_nan_values()
        # Transform the disease specific survival and overall survival to binary
        
        if self.os_threshold != 60:
            
            # save a csv with the old and new values, and the survival time and vital status
            df = pd.DataFrame({'old_overall_survival': self.clinical_data['overall_survival'], 'new_overall_survival': (self.clinical_data['survival_time'] < self.os_threshold).astype(int), 'survival_time': self.clinical_data[cancer_id]['survival_time'], 'vital_status': self.clinical_data[cancer_id]['vital_status']})
            
            self.clinical_data['overall_survival'] = (self.clinical_data['survival_time'] < self.os_threshold).astype(int)
            
            df['new_overall_survival'] = self.clinical_data['overall_survival']  
            
            df.to_csv('check_survival_time.csv', index=False)
        else:            
            
            
            self.clinical_data['disease_specific_survival'] = (self.clinical_data['disease_specific_survival'] < self.os_threshold).astype(int)        
            self.clinical_data['overall_survival'] = (self.clinical_data['overall_survival'] < self.os_threshold).astype(int) #target
        

    def normalize_clinical_data(self):
        self.logger.info('Normalize clinical numerical data using all samples')
        # Impute the missing values with mean
        #['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        self.preprocess_clinical_numeric_data()        
        # CATEGORICAL COLS
        self.clinical_data = pd.get_dummies(self.clinical_data, columns=self.chosen_clinical_categorical_ids, dtype=float)  
        # check that "gender_male" is still present 
        
        self.clinical_data = self.clinical_data.select_dtypes(exclude=['object'])

        # rename columns to be the same as in TCGA dataset
        # 'age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth', 'gender_female', 'gender_male', 
        # 'race_american indian or alaska native', 'race_asian', 'race_black or african american', 
        # 'race_not reported', 'race_white', 'ethnicity_hispanic or latino', 
        # 'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'race_native hawaiian or other pacific islander'
        #change all columns to lower case
        self.clinical_data.columns = map(str.lower, self.clinical_data.columns)
        print(self.clinical_data.columns)

        self.clinical_data.rename({'race_0.0':'race_not reported', 
                                   'race_1.0':'race_white', 'race_2.0':'race_asian', 'ethnicity_0.0': 'ethnicity_not reported', 
                                   'ethnicity_1.0':'ethnicity_not hispanic or latino', 'ethnicity_2.0': 'ethnicity_hispanic or latino' }, inplace=True, axis=1)
        

        if "gender_female" not in self.clinical_data.columns:
            self.clinical_data["gender_female"] = 0
        if "gender_male" not in self.clinical_data.columns:
                    # 0 if gender_female is 1, 1 if gender_female is 0
                    self.clinical_data['gender_male'] = 1 - self.clinical_data['gender_female']
        
        if "race_asian" not in self.clinical_data.columns:
             self.clinical_data["race_asian"] = 0
        if "race_white" not in self.clinical_data.columns:
             self.clinical_data['race_white'] = 0
        if "race_black or african american" not in self.clinical_data.columns:
            self.clinical_data['race_black or african american'] =0
        if "race_not reported" not in self.clinical_data.columns:
             self.clinical_data["race_not reported"] =0
        if "ethnicity_not reported" not in self.clinical_data.columns:
             self.clinical_data["ethnicity_not reported"] =0
        if "race_american indian or alaska native" not in self.clinical_data.columns:
            self.clinical_data['race_american indian or alaska native'] =0
       
        if "ethnicity_hispanic or latino" not in self.clinical_data.columns:
            self.clinical_data['ethnicity_hispanic or latino'] = 0
        if "race_native hawaiian or other pacific islander" not in self.clinical_data.columns:
            self.clinical_data['race_native hawaiian or other pacific islander'] = 0

        # assert that at least one of the race_ columns is 1
        assert self.clinical_data[['race_native hawaiian or other pacific islander','race_american indian or alaska native', 'race_asian', 'race_black or african american',
        'race_not reported', 'race_white']].sum(axis=1).min() == 1


        # assert that at least one of the ethnicity_ columns is 1
        assert self.clinical_data[['ethnicity_hispanic or latino', 
        'ethnicity_not hispanic or latino', 'ethnicity_not reported']].sum(axis=1).min() == 1
                                                 

        #assigned directly so that the order is preserved
        self.all_clinical_feature_ids = ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth', 
        'gender_female', 'gender_male', 'race_american indian or alaska native', 'race_asian', 'race_black or african american',
        'race_not reported', 'race_white', 'ethnicity_hispanic or latino', 
        'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'race_native hawaiian or other pacific islander']

    def log_data_info(self):
                # Log the information of the dataset.
        
        self.logger.info('External DS - Batch size {}'.format(self.batch_size))
        self.logger.info('External DS - Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.patient_ids), len(self.genomic_features), len(self.clinical_features)
        ))
        self.logger.info('External DS - Target Type {}'.format(self.target_type)) #Target Type overall_survival
       
        
    def concat_data(self):
        # Concatenate the genomic and clinical data , having the genes and clinical features as columns   
        #save clinical data to a csv file to check the nan values
        self.clinical_data.to_csv('clin_data.csv', index=True)  
               
        self.data = pd.merge(self.clinical_data, self.genomic_data , left_index=True, right_index=True)
        
        self.data['project_id'] =self.project_id_task_descriptor

        #get rid of object type columns if present
        self.data = self.data.select_dtypes(exclude=['object'])       
        
        self.logger.info('External DS - Total {} samples'.format(len(self.data)))
        self.logger.info('External DS - Total {} features'.format(len(self.data.columns)))

        self.logger.info('External DS - Overall survival imbalance ratio {} %'.format(
            sum(self.data['overall_survival']) / len(self.data['overall_survival']) * 100
        ))
        
        #check if there are any missing values
        #self.logger.info('External DS - Total {} missing values'.format(self.data.isnull().sum().sum()))
        # save the data to a csv file to check the nan values
        #self.data.to_csv('format_ext_data.csv', index=True)

    

    def setup(self, only_test = True):
     
        #this dataset is only for testing 
        self.prepare_data()
        self.get_chosen_features(self.chosen_features)
        self.normalize_clinical_data()
        self.concat_data()
        self.split_data(only_test = only_test)
        self.create_tensors()

    def create_tensors(self):
                   
        #get rid of the ID column
        self.genomic_data = self.genomic_data.drop(columns=['Unnamed: 0'])
        self._genomics = torch.tensor(self.genomic_data.values, dtype=torch.float32)
        self._clinicals = torch.tensor(self.clinical_data.values, dtype=torch.float32)
        self.targets = torch.tensor(self.overall_survivals.values, dtype=torch.float32)
        self._survival_times = torch.tensor(self.disease_specific_survivals.values, dtype=torch.float32)
        self._vital_statuses = torch.tensor(self.vital_status.values, dtype=torch.float32)

    def split_data(self, only_test = True):
            # Split the data into train, validation, and test sets
            if not only_test:
                self.logger.info('Splitting data into train and test sets...')
                train_data, test_data = [], []
                for project_id in self.project_id:
                    project_data = self.data
                    project_data = project_data.sample(frac=1) #shuffle
                    num_samples = len(project_data)
                    num_train_samples = int(num_samples * 0.8)
                    train_data.append(project_data.iloc[:num_train_samples])
                    test_data.append(project_data.iloc[num_train_samples:])
                self.train_data = pd.concat(train_data)
                
                self.test_data = pd.concat(test_data)
            else:
                self.logger.info('Splitting data into test set...')
                # (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
                self.test_data = self.data
        


    def DataLoader(self, data, shuffle=False, drop_last=False):
        
        dataset = CustomDataset(data=data, genomic_features=self.genomic_features, clinical_features=self.all_clinical_feature_ids)
        print("clinical features", self.all_clinical_feature_ids)   
        # Create a DataLoader from the TensorDataset
        sampler = RandomSampler( data_source=dataset, replacement=True, num_samples=len(dataset))   
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=default_collate,
            pin_memory=True,
            sampler=sampler,
            drop_last=drop_last

        )
        return dataloader
    

    
    def get_dataset(self):
        dataset = CustomDataset(data=self.test_data, genomic_features=self.genomic_features, clinical_features=self.all_clinical_feature_ids)
        return dataset
    
    def bootstrap_test_dataloader(self, dataset):
        # Create a DataLoader from the CustomDataset
        sampler = RandomSampler( data_source=dataset, replacement=True, num_samples=len(dataset))   
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_collate,
            pin_memory=True,
            sampler=sampler,
            drop_last=False
        )
        return dataloader
         


    def train_dataloader(self):
        return self.DataLoader(self.train_data, shuffle=False, drop_last=True)

    def val_dataloader(self):
        return self.DataLoader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self.DataLoader(self.test_data, shuffle=False )


    def teardown(self, stage=None):
        # Clean up any resources used by the data module
        if stage == 'fit' or stage is None:
            shutil.rmtree(self.cache_directory)



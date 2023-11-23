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
            
            index = index#row['PATIENT_ID']
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
        # Download the necessary data files
        # load sclc_ucologne_2015 data
        self.genomic_data = pd.read_csv(self.data_dir + '/data_mrna_seq_tpm_small.csv', header=0, sep=',')
       
        self.filter_genes()

        self.clinical_data = pd.read_csv(self.data_dir + '/data_clinical_patient.csv', header=0, sep=',')
        #PATIENT_ID	gender	ethnicity	race	year_of_diagnosis	year_of_birth	
        # overall_survival	vital_status	disease_specific_survival	primary_site
        #(genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        #print(self.clinical_data.columns)
        self.overall_survivals = self.clinical_data.overall_survival
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival']
        self.primary_sites = self.clinical_data['primary_site']
        self.vital_status = self.clinical_data['vital_status']
        

        
    def filter_genes(self):
        self.genomic_data = self.genomic_data[['Unnamed: 0']+ self.chosen_genes ]


   
    
    def get_patient_ids(self):
        # Get the patient IDs
        self.patient_ids = np.unique(self.clinical_data['PATIENT_ID'])
       
        self.logger.info('Total {} patients'.format(len(self.patient_ids)))

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
        #the std is 0 for year_of_diagnosis, all samples were taken in 2015
        self.clinical_data[self.chosen_clinical_numerical_ids] /= clinical_std

        #remove nan values from vital status, replace nan values with 0
        self.clinical_data['vital_status'] = self.clinical_data['vital_status'].fillna(0)
        self.clinical_data['survival_time'] = self.clinical_data['disease_specific_survival']

        # Transform the disease specific survival and overall survival to binary
        months_threshold = 60 # 5 years 
        self.clinical_data['disease_specific_survival'] = (self.clinical_data['disease_specific_survival'] >= months_threshold).astype(int)
        self.clinical_data['overall_survival'] = (self.clinical_data['overall_survival'] >= months_threshold).astype(int)

        self.overall_survivals = self.clinical_data['overall_survival'] 
        self.disease_specific_survivals = self.clinical_data['disease_specific_survival'] 
        self.vital_status = self.clinical_data['vital_status']

    def normalize_clinical_data(self):
        self.logger.info('Normalize clinical numerical data using all samples')
        # Impute the missing values with mean
        #['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth']
        
        self.preprocess_clinical_numeric_data()        

    
        # CATEGORICAL COLS

        self.clinical_data = pd.get_dummies(self.clinical_data, columns=self.chosen_clinical_categorical_ids, dtype=float)             

        self.clinical_data = self.clinical_data.select_dtypes(exclude=['object'])

        # rename columns to be the same as in TCGA dataset
        # 'age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth', 'gender_female', 'gender_male', 
        # 'race_american indian or alaska native', 'race_asian', 'race_black or african american', 
        # 'race_not reported', 'race_white', 'ethnicity_hispanic or latino', 
        # 'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'race_native hawaiian or other pacific islander'
        self.clinical_data.rename({'gender_Female': 'gender_female', 'gender_Male': 'gender_male', 'race_0.0':'race_not reported', 
                                   'race_1.0':'race_white', 'race_2.0':'race_asian', 'ethnicity_0.0': 'ethnicity_not reported', 'ethnicity_1.0':'ethnicity_not hispanic or latino' }, inplace=True, axis=1)
        
        #add the binary columns: 'race_american indian or alaska native', 'race_black or african american', 'ethnicity_hispanic or latino'

        self.clinical_data['race_american indian or alaska native'] =0
        self.clinical_data['race_black or african american'] =0
        self.clinical_data['ethnicity_hispanic or latino'] = 0
        self.clinical_data['race_native hawaiian or other pacific islander'] = 0

        #assigned directly so that the order is preserved
        self.all_clinical_feature_ids = ['age_at_diagnosis', 'year_of_diagnosis', 'year_of_birth', 
        'gender_female', 'gender_male', 'race_american indian or alaska native', 'race_asian', 'race_black or african american',
        'race_not reported', 'race_white', 'ethnicity_hispanic or latino', 
        'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'race_native hawaiian or other pacific islander']

        

       
        
   
        
       

        

    def log_data_info(self):
                # Log the information of the dataset.
        self.logger.info('Creating a TCGA Program Dataset with {} Projects...'.format(len(self.project_id)))
        self.logger.info('Batch size {}'.format(self.batch_size))
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(
            len(self.patient_ids), len(self.genomic_features), len(self.clinical_features)
        ))
        self.logger.info('Target Type {}'.format(self.target_type)) #Target Type overall_survival
        # self.logger.info('Overall survival imbalance ratio {} %'.format(
        #     sum(self.overall_survivals) / len(self.overall_survivals) * 100
        # ))
        # self.logger.info('Number of survivals {}, of a total of{}'.format(  sum( self.overall_survivals), len(self.overall_survivals)))
        
    def concat_data(self):
        # Concatenate the genomic and clinical data , having the genes and clinical features as columns       
        
        
        self.data = pd.merge(self.clinical_data, self.genomic_data , left_index=True, right_index=True)

        

        
        # fill all project ids with self.project_id
        
        self.data['project_id'] =self.project_id_task_descriptor #temporary value

        #get rid of object type columns
        self.data = self.data.select_dtypes(exclude=['object'])       
        
        self.logger.info('Total {} samples'.format(len(self.data)))
        self.logger.info('Total {} features'.format(len(self.data.columns)))

        self.logger.info('Overall survival imbalance ratio {} %'.format(
            sum(self.data['overall_survival']) / len(self.data['overall_survival']) * 100
        ))
        #self.logger.info('Number of survivals {}, of a total of {}'.format(  sum( self.data['overall_survival']), len(self.data['overall_survival'])))

        #check if there are any missing values
        self.logger.info('Total {} missing values'.format(self.data.isnull().sum().sum()))
        # save the data to a csv file to check the nan values
        self.data.to_csv('format_ext_data.csv', index=True)

    

    def setup(self, stage=None):
        # Load the data files and split them into train, validation, and test sets
        #this dataset is only for testing 
        self.prepare_data()
        self.get_chosen_features(self.chosen_features)
        
        self.normalize_clinical_data()
        self.concat_data()
        self.split_data()
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
        dataset = CustomDataset(data=data, genomic_features=self.genomic_features, clinical_features=self.all_clinical_feature_ids)
        # Create a DataLoader from the TensorDataset

        sampler = RandomSampler( data_source=dataset, replacement=True, num_samples=len(dataset))   


        dataloader = DataLoader(dataset, batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=default_collate,
            pin_memory=True,
            sampler=sampler
        )


        return dataloader


    def train_dataloader(self):
        return self.DataLoader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.DataLoader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self.DataLoader(self.test_data, shuffle=False, )


    def teardown(self, stage=None):
        # Clean up any resources used by the data module
        if stage == 'fit' or stage is None:
            shutil.rmtree(self.cache_directory)



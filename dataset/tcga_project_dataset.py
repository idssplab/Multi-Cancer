import math
import numpy as np
import pandas as pd
from datetime import datetime
from preprocess import TCGA_Project
from base import BaseDataset
from utils.logger import get_logger
from utils.util import check_cache_files
from pathlib import Path


class TCGA_Project_Dataset(BaseDataset):
    '''
    TCGA Project Dataset
    '''
    def __init__(self, project_id, data_directory, cache_directory, chosen_features=dict(), well_known_gene_ids=None, genomic_type='tpm', target_type='overall_survival', n_threads=1):
        '''
        Initialize the TCGA Project Dataset with parameters.

        Needed parameters
        :param project_id: Specify the project id.
        :param data_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param chosen_features: Specify the features that you want to use.
        :param well_known_gene_ids: The well-known gene ids of the project.
        :param genomic_type: The genomic type that uses in this project.
        :param target_type: The target type that uses in the project.
        :param n_threads: The number of threads to user for concatenating genomic data.
        '''
        self.project_id = project_id

        #Logger
        self.logger = get_logger('preprocess.tcga_project_dataset')
        self.logger.info('Creating a {} Project Dataset...'.format(self.project_id))

        # Directory for data and cache files
        self.data_directory = Path(data_directory)
        self.cache_directory = Path(cache_directory)

        # Get chosen features
        self.chosen_gene_ids = chosen_features.get('gene_ids', [])
        self.chosen_clinical_numerical_ids = chosen_features.get('clinical_numerical_ids', [])
        self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

        # Create TCGA_Project instance
        self.tcga_project_init_kwargs = {
            'project_id': project_id,
            'download_directory': data_directory,
            'cache_directory': cache_directory,
            'well_known_gene_ids': well_known_gene_ids,
            'genomic_type': genomic_type,
            'n_threads': n_threads
        }
        self.tcga_project = TCGA_Project(**self.tcga_project_init_kwargs)

        # Specify the target type
        self.target_type = target_type

        # Get data from TCGA_Project instance
        self._genomics, self._clinicals, self._vital_statuses, self._overall_survivals, self._disease_specific_survivals, self._survival_times, self._patient_ids, self._genomic_ids, self._clinical_ids = self._getdata()
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(len(self.targets[self.targets >= 0]), len(self.genomic_ids), len(self.clinical_ids)))
        self.logger.info('Target Type {}'.format(self.target_type))
        self.logger.info('Overall survival imbalance ratio {} %'.format(sum(self.overall_survivals)/len(self.overall_survivals)*100))
        self.logger.info('Disease specific survival event rate {} %'.format(sum(self.disease_specific_survivals >= 0)/len(self.disease_specific_survivals)*100))
        self.logger.info('Disease specific survival imbalance ratio {} %'.format(
            sum(self.disease_specific_survivals[self.disease_specific_survivals >= 0])/len(self.disease_specific_survivals[self.disease_specific_survivals >= 0])*100)
        )

        # Initialize BaseDataset instance
        self.base_dataset_init_kwargs = {
            'data_root_directory': data_directory,
            'cache_root_directory': cache_directory
        }
        super().__init__(**self.base_dataset_init_kwargs)

    def _getdata(self):
        '''
        Get the data from TCGA Project
        '''
        df_genomic = self.tcga_project.genomic.T
        if self.chosen_gene_ids not in ['ALL']:
            df_genomic = df_genomic[self.chosen_gene_ids]

        df_clinical = self.tcga_project.clinical.T[self.chosen_clinical_ids]

        if len(self.chosen_clinical_numerical_ids):
            df_clinical[self.chosen_clinical_numerical_ids] = df_clinical[self.chosen_clinical_numerical_ids].astype('float64')

            indices_latest_file_path = check_cache_files(
                cache_directory=self.cache_directory,
                regex=f'indices_*'
            )

            if indices_latest_file_path:
                indices_latest_file_created_date = indices_latest_file_path.name.split('.')[0].split('_')[-1]
                self.logger.info('Using indices cache files created at {} from {}'.format(
                    datetime.strptime(indices_latest_file_created_date, "%Y%m%d%H%M%S"),
                    self.cache_directory
                ))
                self.logger.info('Normalize clinical numerical data using training samples only')

                indices_cache = np.load(indices_latest_file_path)
                train_indices = indices_cache['train']

                clinical_numerical_ids_mean = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].mean()
                clinical_numerical_ids_std = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].std()
            else:
                self.logger.info('Normalize clinical numerical data using all samples')
                clinical_numerical_ids_mean = df_clinical[self.chosen_clinical_numerical_ids].mean()
                clinical_numerical_ids_std = df_clinical[self.chosen_clinical_numerical_ids].std()

            # Impute the missing values with mean
            df_clinical[self.chosen_clinical_numerical_ids] = df_clinical[self.chosen_clinical_numerical_ids].fillna(clinical_numerical_ids_mean.to_dict())

            # Normalize the numerical values
            df_clinical[self.chosen_clinical_numerical_ids] = (df_clinical[self.chosen_clinical_numerical_ids] - clinical_numerical_ids_mean) / clinical_numerical_ids_std

        if len(self.chosen_clinical_categorical_ids):
            df_clinical = pd.get_dummies(df_clinical, columns=self.chosen_clinical_categorical_ids)

            all_tcga_clinical_categorical_ids_latest_file_path = check_cache_files(
                cache_directory=self.cache_directory,
                regex=f'all_tcga_clinical_categorical_ids_*'
            )
            if all_tcga_clinical_categorical_ids_latest_file_path:
                all_tcga_clinical_categorical_ids_latest_file_created_date = all_tcga_clinical_categorical_ids_latest_file_path.name.split('.')[0].split('_')[-1]
                self.logger.info('Using all tcga clinical categorical ids cache files created at {} for {}'.format(
                    datetime.strptime(all_tcga_clinical_categorical_ids_latest_file_created_date, "%Y%m%d%H%M%S"),
                    self.project_id
                ))

                df_all_tcga_clinical_categorical_ids = pd.read_csv(all_tcga_clinical_categorical_ids_latest_file_path, sep='\t', header=None).squeeze()
                all_tcga_clinical_ids = self.chosen_clinical_numerical_ids+df_all_tcga_clinical_categorical_ids.squeeze().tolist()
                df_clinical = df_clinical.reindex(columns=all_tcga_clinical_ids).fillna(0)

        df_vital_status = self.tcga_project.vital_status.T
        df_overall_survival = self.tcga_project.overall_survival.T
        df_disease_specific_survival = self.tcga_project.disease_specific_survival.T
        df_survival_time = self.tcga_project.survival_time.T

        df_total = pd.concat([df_genomic, df_clinical, df_vital_status, df_overall_survival, df_disease_specific_survival, df_survival_time], axis=1)

        genomics = df_total[df_genomic.columns].to_numpy()
        clinicals = df_total[df_clinical.columns].to_numpy()
        vital_statuses = df_total[df_vital_status.columns].squeeze().to_numpy()
        overall_survivals = df_total[df_overall_survival.columns].squeeze().to_numpy()
        disease_specific_survivals = df_total[df_disease_specific_survival.columns].squeeze().to_numpy()
        survival_times = df_total[df_survival_time.columns].squeeze().to_numpy()
        patient_ids = tuple(df_total.index.to_list())
        genomic_ids = tuple(df_genomic.columns.to_list())
        clinical_ids = tuple(df_clinical.columns.to_list())

        return genomics, clinicals, vital_statuses, overall_survivals, disease_specific_survivals, survival_times, patient_ids, genomic_ids, clinical_ids

    def __getitem__(self, index):
        '''
        Support the indexing of the dataset
        '''
        return (self._genomics[index], self._clinicals[index], index), (self.targets[index], self._survival_times[index], self._vital_statuses[index])

    def __len__(self):
        '''
        Return the size of the dataset
        '''
        return len(self.targets)

    @property
    def data(self):
        '''
        Return the genomic and clinical data.
        '''
        return np.hstack((self._genomics, self._clinicals))

    @property
    def genomics(self):
        '''
        Return the genomic data.
        '''
        return self._genomics

    @property
    def clinicals(self):
        '''
        Return the clinical data.
        '''
        return self._clinicals

    @property
    def vital_statuses(self):
        '''
        Return the vital status data.
        '''
        return self._vital_statuses

    @property
    def overall_survivals(self):
        '''
        Return the 5 year overall survival data.
        '''
        return self._overall_survivals

    @property
    def disease_specific_survivals(self):
        '''
        Return the 5 year disease specific survival data.
        '''
        return self._disease_specific_survivals

    @property
    def survival_times(self):
        '''
        Return the survival time data.
        '''
        return self._survival_times

    @property
    def weights(self):
        '''
        Return the weights for each data.
        '''
        weights = np.zeros_like(self.targets, dtype='float64')
        for i in range(self.targets.min(), self.targets.max()+1):
            weights[self.targets == i] = math.sqrt(1.0 / (self.targets == i).sum())
        return weights

    @property
    def stratified_targets(self):
        '''
        Return the target data according to target_type.
        '''
        return np.abs(self._overall_survivals - 1)

    @property
    def targets(self):
        '''
        Return the target data according to target_type.
        '''
        if self.target_type == 'overall_survival':
            return self._overall_survivals
        elif self.target_type == 'disease_specific_survival':
            return self._disease_specific_survivals
        else:
            raise KeyError(f'Wrong target type')

    @property
    def patient_ids(self):
        '''
        Return the patient ids
        '''
        return self._patient_ids
    
    @property
    def genomic_ids(self):
        '''
        Return the genomic ids
        '''
        return self._genomic_ids

    @property
    def clinical_ids(self):
        '''
        Return the clinical ids
        '''
        return self._clinical_ids
import math
import numpy as np
import pandas as pd
from datetime import datetime
from preprocess import TCGA_Project
from base import BaseDataset
from utils.logger import get_logger
from utils.api import get_filters_result_from_project
from utils.util import check_cache_files
from pathlib import Path


PROJECT_MAX_COUNT = 1000


class TCGA_Program_Dataset(BaseDataset):
    '''
    TCGA Program Dataset
    '''
    def __init__(self, project_ids, data_directory, cache_directory, chosen_features=dict(), genomic_type='tpm', target_type='overall_survival', n_threads=1):
        '''
        Initialize the TCGA Program Dataset with parameters.

        Needed parameters
        :param project_id: Specify the project id.
        :param data_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param chosen_features: Specify the features that you want to use.
        :param genomic_type: The genomic type that uses in this project.
        :param target_type: Specify the wanted target type that you want to use.
        :param n_threads: The number of threads to user for concatenating genomic data.
        '''
        if project_ids not in ['ALL']:
            self.project_ids = project_ids
        else:
            project_filters = {
                '=': {'program.name': 'TCGA'}
            }
            self.project_ids = [
                project_metadata['id'] for project_metadata in get_filters_result_from_project(
                    filters=project_filters,
                    sort='summary.case_count:desc',
                    size=PROJECT_MAX_COUNT
                )
            ]

        #Logger
        self.logger = get_logger('preprocess.tcga_program_dataset')
        self.logger.info('Creating a TCGA Program Dataset with {} Projects...'.format(len(self.project_ids)))

        # Directory for data and cache files
        self.data_directory = Path(data_directory)
        self.cache_directory = Path(cache_directory)

        # Get chosen features
        self.chosen_gene_counts = chosen_features.get('gene_counts', 20)
        self.chosen_project_gene_ids = chosen_features.get('gene_ids', {})
        for project_id in self.project_ids:
            if project_id not in self.chosen_project_gene_ids:
                if project_ids not in ['ALL']:
                    raise ValueError(f'Gene ids is null for {project_id}')
                else:
                    self.chosen_project_gene_ids[project_id] = []
                    self.logger.debug(f'Gene ids is null for {project_id}')
        self.chosen_clinical_numerical_ids = chosen_features.get('clinical_numerical_ids', [])
        self.chosen_clinical_categorical_ids = chosen_features.get('clinical_categorical_ids', [])
        self.chosen_clinical_ids = self.chosen_clinical_numerical_ids + self.chosen_clinical_categorical_ids

        # Create TCGA_Project instances
        self.tcga_projects = {}
        for project_id in self.project_ids:
            self.tcga_project_init_kwargs = {
                'project_id': project_id,
                'download_directory': self.data_directory.joinpath(project_id),
                'cache_directory': self.cache_directory.joinpath(project_id),
                'genomic_type': genomic_type,
                'n_threads': n_threads
            }

            self.tcga_projects[project_id] = TCGA_Project(**self.tcga_project_init_kwargs)

        # Specify the target type
        self.target_type = target_type

        # Get data from TCGA_Project instance
        self._genomics, self._clinicals, self._vital_statuses, self._overall_survivals, self._disease_specific_survivals, self._survival_times, self._primary_sites, self._project_ids, self._primary_site_ids, self._patient_ids, self._genomic_ids, self._clinical_ids = self._getdata()
        self.logger.info('Total {} patients, {} genomic features and {} clinical features'.format(len(self.patient_ids), len(self.genomic_ids), len(self.clinical_ids)))
        self.logger.info('Target Type {}'.format(self.target_type))
        self.logger.info('Overall survival imbalance ratio {} %'.format(sum(self.overall_survivals)/len(self.overall_survivals)*100))
        self.logger.info('Disease specific survival event rate {} %'.format(sum(self.disease_specific_survivals >= 0)/len(self.disease_specific_survivals)*100))
        self.logger.info('Disease specific survival imbalance ratio {} %'.format(
            sum(self.disease_specific_survivals[self.disease_specific_survivals >= 0])/len(self.disease_specific_survivals[self.disease_specific_survivals >= 0])*100)
        )
        self.logger.info('{} kinds of primary sites {}'.format(len(np.unique(self.primary_sites)), ' / '.join(self.primary_site_ids)))

        # Initialize BaseDataset instance
        self.base_dataset_init_kwargs = {
            'data_root_directory': data_directory,
            'cache_root_directory': cache_directory
        }
        super().__init__(**self.base_dataset_init_kwargs)

    def _getdata(self):
        '''
        Get the data from TCGA Program
        '''
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

        for project_id, tcga_project in self.tcga_projects.items():
            df_genomic = tcga_project.genomic.T
            if self.chosen_project_gene_ids[project_id] not in ['ALL']:
                df_genomic = df_genomic[self.chosen_project_gene_ids[project_id]]
            else:
                raise ValueError(f'No gene ids specified for {project_id}')

            # Rename the gene ids to numbers
            df_genomic.rename(columns={column_name: index for index, column_name in enumerate(df_genomic.columns)}, inplace=True)

            df_clinical = tcga_project.clinical.T[self.chosen_clinical_ids]

            if len(self.chosen_clinical_numerical_ids):
                df_clinical[self.chosen_clinical_numerical_ids] = df_clinical[self.chosen_clinical_numerical_ids].astype('float64')

                indices_latest_file_path = check_cache_files(
                    cache_directory=self.cache_directory.joinpath(project_id),
                    regex=f'indices_*'
                )

                if indices_latest_file_path:
                    indices_latest_file_created_date = indices_latest_file_path.name.split('.')[0].split('_')[-1]
                    self.logger.info('Using indices cache files created at {} from {}'.format(
                        datetime.strptime(indices_latest_file_created_date, "%Y%m%d%H%M%S"),
                        self.cache_directory.joinpath(project_id)
                    ))
                    self.logger.info('Normalize clinical numerical data using training samples only')

                    indices_cache = np.load(indices_latest_file_path)
                    train_indices = indices_cache['train']
                    test_indices = indices_cache['test']

                    clinical_numerical_ids_mean = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].mean()
                    clinical_numerical_ids_std = df_clinical[self.chosen_clinical_numerical_ids].iloc[train_indices].std()

                    train_patient_ids.extend(df_clinical.iloc[train_indices].index.to_list())
                    test_patient_ids.extend(df_clinical.iloc[test_indices].index.to_list())
                else:
                    self.logger.info('Normalize clinical numerical data using all samples')
                    clinical_numerical_ids_mean = df_clinical[self.chosen_clinical_numerical_ids].mean()
                    clinical_numerical_ids_std = df_clinical[self.chosen_clinical_numerical_ids].std()

                    train_patient_ids.extend(df_clinical.index.to_list())

                # Impute the missing values with mean
                df_clinical[self.chosen_clinical_numerical_ids] = df_clinical[self.chosen_clinical_numerical_ids].fillna(clinical_numerical_ids_mean.to_dict())

                # Normalize the numerical values
                df_clinical[self.chosen_clinical_numerical_ids] = (df_clinical[self.chosen_clinical_numerical_ids] - clinical_numerical_ids_mean) / clinical_numerical_ids_std

            if len(self.chosen_clinical_categorical_ids):
                df_clinical = pd.get_dummies(df_clinical, columns=self.chosen_clinical_categorical_ids)

                all_tcga_clinical_categorical_ids_latest_file_path = check_cache_files(
                    cache_directory=self.cache_directory.joinpath(project_id),
                    regex=f'all_tcga_clinical_categorical_ids_*'
                )
                if all_tcga_clinical_categorical_ids_latest_file_path:
                    all_tcga_clinical_categorical_ids_latest_file_created_date = all_tcga_clinical_categorical_ids_latest_file_path.name.split('.')[0].split('_')[-1]
                    self.logger.info('Using all tcga clinical categorical ids cache files created at {} for {}'.format(
                        datetime.strptime(all_tcga_clinical_categorical_ids_latest_file_created_date, "%Y%m%d%H%M%S"),
                        project_id
                    ))

                    df_all_tcga_clinical_categorical_ids = pd.read_csv(all_tcga_clinical_categorical_ids_latest_file_path, sep='\t', header=None).squeeze()
                    all_tcga_clinical_ids = self.chosen_clinical_numerical_ids+df_all_tcga_clinical_categorical_ids.squeeze().tolist()
                    df_clinical = df_clinical.reindex(columns=all_tcga_clinical_ids).fillna(0)

            df_vital_status = tcga_project.vital_status.T
            df_overall_survival = tcga_project.overall_survival.T
            df_disease_specific_survival = tcga_project.disease_specific_survival.T
            df_survival_time = tcga_project.survival_time.T
            df_primary_site = tcga_project.primary_site.T
            df_project_id = pd.DataFrame(data=[self.project_ids.index(project_id)]*len(df_primary_site), index=df_primary_site.index, columns=['project_id'])
            
            df_genomics.append(df_genomic)
            df_clinicals.append(df_clinical)
            
            df_vital_statuses.append(df_vital_status)
            df_overall_survivals.append(df_overall_survival)
            df_disease_specific_survivals.append(df_disease_specific_survival)
            df_survival_times.append(df_survival_time)
            df_primary_sites.append(df_primary_site)
            df_project_ids.append(df_project_id)

        df_genomics = pd.concat(df_genomics)
        df_clinicals = pd.concat(df_clinicals).fillna(0)
        df_vital_statuses = pd.concat(df_vital_statuses)
        df_overall_survivals = pd.concat(df_overall_survivals)
        df_disease_specific_survivals = pd.concat(df_disease_specific_survivals)
        df_survival_times = pd.concat(df_survival_times)
        df_primary_sites = pd.concat(df_primary_sites).astype('category')
        df_project_ids = pd.concat(df_project_ids)

        if len(self.project_ids) == len(get_filters_result_from_project(filters={'=': {'program.name': 'TCGA'}}, size=PROJECT_MAX_COUNT)):
            all_tcga_clinical_categorical_columns = df_clinicals.columns[~df_clinicals.columns.isin(self.chosen_clinical_numerical_ids)].to_series()
            for project_id in self.project_ids:
                all_tcga_clinical_categorical_ids_latest_file_path = check_cache_files(
                    cache_directory=self.cache_directory.joinpath(project_id),
                    regex=f'all_tcga_clinical_categorical_ids_*'
                )

                if all_tcga_clinical_categorical_ids_latest_file_path:
                    all_tcga_clinical_categorical_ids_cache = pd.read_csv(all_tcga_clinical_categorical_ids_latest_file_path, sep='\t', header=None).squeeze()
                    if not len(set(all_tcga_clinical_categorical_columns) - set(all_tcga_clinical_categorical_ids_cache)):
                        continue

                all_tcga_clinical_categorical_columns.to_csv(
                    self.cache_directory.joinpath(project_id, f'all_tcga_clinical_categorical_ids_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
                    sep='\t', index=False, header=False
                )
                self.logger.info('Saving all tcga clinical categorical ids to {}'.format(self.cache_directory.joinpath(project_id)))

        df_totals = pd.concat([df_genomics, df_clinicals, df_vital_statuses, df_overall_survivals, df_disease_specific_survivals, df_survival_times, df_primary_sites, df_project_ids], axis=1)

        genomics = df_totals[df_genomics.columns].to_numpy()
        clinicals = df_totals[df_clinicals.columns].to_numpy()
        vital_statuses = df_totals[df_vital_statuses.columns].squeeze().to_numpy()
        overall_survivals = df_totals[df_overall_survivals.columns].squeeze().to_numpy()
        disease_specific_survivals = df_totals[df_disease_specific_survivals.columns].squeeze().to_numpy()
        survival_times = df_totals[df_survival_times.columns].squeeze().to_numpy()
        primary_sites = df_totals[df_primary_sites.columns].squeeze().cat.codes.to_numpy()
        project_ids = df_totals[df_project_ids.columns].squeeze().to_numpy()
        primary_site_ids = tuple(df_totals[df_primary_sites.columns].squeeze().cat.categories.to_list())
        patient_ids = tuple(df_totals.index.to_list())
        genomic_ids = tuple(df_genomics.columns.to_list())
        clinical_ids = tuple(df_clinicals.columns.to_list())

        indices = {}
        indices['train'] = np.array([index for index, patient_id in enumerate(patient_ids) if patient_id in train_patient_ids])
        indices['test'] = np.array([index for index, patient_id in enumerate(patient_ids) if patient_id in test_patient_ids])

        for file_path in self.cache_directory.glob('indices_*'):
            if file_path.is_file():
                file_path.unlink()
                self.logger.debug('Removing redundant indices file {}'.format(file_path))

        np.savez(self.cache_directory.joinpath(f'indices_{datetime.now().strftime("%Y%m%d%H%M%S")}.npz'), **indices)
        self.logger.info('Saving train and test indices to {}'.format(self.cache_directory))

        return genomics, clinicals, vital_statuses, overall_survivals, disease_specific_survivals, survival_times, primary_sites, project_ids, primary_site_ids, patient_ids, genomic_ids, clinical_ids

    def __getitem__(self, index):
        '''
        Support the indexing of the dataset
        '''
        return (self._genomics[index], self._clinicals[index], index, self._project_ids[index]), (self.targets[index], self._survival_times[index], self._vital_statuses[index])

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
    def primary_sites(self):
        '''
        Return the primary site data.
        '''
        return self._primary_sites

    @property
    def primary_site_ids(self):
        '''
        Return the primary site ids.
        '''
        return self._primary_site_ids

    @property
    def weights(self):
        '''
        Return the weights for each data.
        '''
        weights = np.zeros_like(self._project_ids, dtype='float64')
        for i in range(self._project_ids.min(), self._project_ids.max()+1):
            weights[self._project_ids == i] = math.sqrt(1.0 / (self._project_ids == i).sum())
        return weights

    @property
    def targets(self):
        '''
        Return the target data according to target_type.
        '''
        if self.target_type == 'overall_survival':
            return self._overall_survivals
        elif self.target_type == 'disease_specific_survival':
            return self._disease_specific_survivals
        elif self.target_type == 'primary_site':
            return self._primary_sites
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

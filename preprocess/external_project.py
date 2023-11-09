
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from hdf5storage import savemat, loadmat
import sys
sys.path.insert(0, '/home/zow/prognosis-prediction')
from preprocess.external_dataset import ExternalDataset
from utils.logger import get_logger
from utils.api.tcga_api import (get_metadata_from_project, get_filters_result_from_case, get_filters_result_from_file,
                                download_file, download_files)
from utils.util import check_cache_files


AIC_PREPROCESS = False
AIC_POSTPROCESS = True


class External_Project(object):
    '''
    External Project
    '''
    def __init__(self, project_id, download_directory, cache_directory,
                 well_known_gene_ids=None, genomic_type='tpm', n_threads=1):
        '''
        

        Needed parameters
        :param project_id: Specify the project id.
        :param download_directory: Specify the directory for the downloaded files.
        :param cache_directory: Specify the directory for the cache files.

        Optional parameters
        :param well_known_gene_ids: The well-known gene ids of the project.
        :param genomic_type: Specify the wanted genomic type that you want to use.
        :param n_threads: The number of threads to user for concatenating genomic data.
        '''
        self.project_id = project_id

        # Logger
        self.logger = get_logger('preprocess.tcga_project')

        # Directory for download files
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(exist_ok=True)

        # Directory for cache files
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)

        # Maximum number of threads
        self.n_threads = n_threads

        # create safeguard for external dataset
        
        print('SCLC - UCologne 2015 is an external dataset')
        # Get metadata
        self.project_metadata = ""#self._get_project_metadata(project_id=self.project_id)
        

        # Download files
        self.cases_file_paths = 'Data/sclc_ucologne_2015/data_clinical_patient.tsv'

        self.clinical_data_df = pd.read_csv(self.cases_file_paths, sep='\t', index_col='PATIENT_ID')
        self.genetic_data_df = pd.read_csv('Data/sclc_ucologne_2015/data_mrna_seq_tpm.tsv', sep='\t')


        #get case ids from the index of the dataframe

        self.case_metadatas = {case_id: {} for case_id in self.clinical_data_df.index.to_list()}
        # Sorted case_ids
        self.case_ids = sorted(self.case_metadatas)

        
        

        # Data types
        self.genomic_type = genomic_type

        # Create cases
        self.cases = self._create_cases(
            case_ids=self.case_ids,
            directory=self.download_directory,
            case_metadatas=self.case_metadatas,
            cases_file_paths=self.cases_file_paths,
            genomic_type=self.genomic_type
        )

        # Genomic data
        self._genomic = self._concat_genomic_data(
            cases=self.cases,
            cache_directory=self.cache_directory,
            n_threads=self.n_threads
        )

      
        self.well_known_gene_ids = well_known_gene_ids


        # Clinical data
        self._clinical = self._concat_clinical_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Vital status data
        self._vital_status = self._concat_vital_status_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Survival data
        self._overall_survival = self._concat_overall_survival_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )
        self._disease_specific_survival = self._concat_disease_specific_survival_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )
        self._survival_time = self._concat_survival_time_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

        # Primary site data
        self._primary_site = self._concat_primary_site_data(
            cases=self.cases,
            genomic_data=self._genomic,
            cache_directory=self.cache_directory
        )

    def _get_project_metadata(self, project_id):
        '''
        Get the metadata according to project id from TCGA API.

        :param project_id: Specify the project id.
        '''
        kwargs = {}

        return get_metadata_from_project(project_id=project_id, **kwargs)

    def _get_case_metadatas(self, project_id):
        '''
        Filter the wanted cases and collect their metadatas from TCGA API.

        :param project_id: Specify the project id.
        '''
        case_metadatas = {}

        kwargs = {}
        kwargs['size'] = get_metadata_from_project(project_id, fields=['summary.case_count'])['summary']['case_count']

        # Filter
        kwargs['filters'] = {
            'and': [
                {'=': {'project.project_id': project_id}},
                {'=': {'files.access': 'open'}},
                {'=': {'files.data_type': 'Gene Expression Quantification'}},
                {'=': {'files.experimental_strategy': 'RNA-Seq'}},
                {'=': {'files.data_format': 'TSV'}},
                {'or': [
                    {'=': {'demographic.vital_status': 'Alive'}},
                    {'and': [
                        {'=': {'demographic.vital_status': 'Dead'}},
                        {'not': {'diagnoses.days_to_diagnosis': 'missing'}},
                        {'not': {'demographic.days_to_death': 'missing'}}
                    ]}
                ]}
            ]
        }

        # Expand
        kwargs['expand'] = [
            'annotations',
            'diagnoses',
            'diagnoses.annotations',
            'diagnoses.pathology_details',
            'diagnoses.treatments',
            'demographic',
            'exposures',
            'family_histories',
            'files',
            'follow_ups',
            'follow_ups.molecular_tests',
            'samples',
            'samples.annotations',
            'samples.portions',
            'samples.portions.analytes',
            'samples.portions.analytes.aliquots',
            'samples.portions.analytes.aliquots.annotations',
            'samples.portions.analytes.aliquots.center',
            'samples.portions.analytes.annotations',
            'samples.portions.annotations',
            'samples.portions.center',
            'samples.portions.slides',
            'samples.portions.slides.annotations'
        ]

        for case_metadata in get_filters_result_from_case(**kwargs):
            case_metadatas[case_metadata['id']] = case_metadata

        return case_metadatas

    def _get_case_file_metadatas(self, project_id, case_ids):
        '''
        Filter the wanted files and collect their metadatas from TCGA API.

        :param project_id: Specify the project id.
        :param case_ids: Specify the case ids.
        '''
        cases_file_metadatas = {case_id: {} for case_id in case_ids}

        kwargs = {}
        kwargs['size'] = get_metadata_from_project(project_id, fields=['summary.file_count'])['summary']['file_count']

        # Filter
        kwargs['filters'] = {
            'and': [
                {'=': {'cases.project.project_id': project_id}},
                {'=': {'access': 'open'}},
                {'or': [
                    {'and': [
                        {'=': {'data_type': 'Clinical Supplement'}},
                        {'=': {'data_format': 'BCR XML'}}
                    ]},
                    {'and': [
                        {'=': {'data_type': 'Gene Expression Quantification'}},
                        {'=': {'experimental_strategy': 'RNA-Seq'}},
                        {'=': {'data_format': 'TSV'}}
                    ]}
                ]}
            ]
        }

        # Fields
        kwargs['fields'] = ['cases.case_id', 'file_name', 'file_id']

        for case_file_metadata in get_filters_result_from_file(**kwargs):
            file_name = case_file_metadata['file_name']
            file_id = case_file_metadata['file_id']

            if len(case_file_metadata['cases']) == 1:
                case_id = case_file_metadata['cases'][0]['case_id']
            else:
                raise ValueError(f'More than one case for {file_name}')

            if case_id in case_ids:
                cases_file_metadatas[case_id][file_name] = file_id
            else:
                continue

        return cases_file_metadatas



    def _create_cases(self, case_ids, directory, case_metadatas, cases_file_paths, genomic_type):
        '''
        Create the TCGA Case instance according to case ids and directory.

        :param case_ids: Specify the case ids.
        :param directory: Specify the root directory for the project.
        :param case_metadatas: The metadata of the cases.
        :param case_file_paths: The related file paths of the cases.
        :param genomic_type: Specify the wanted genomic type that you want to use.
        '''
        self.logger.info('Creating {} cases for {}...'.format(len(case_ids), self.project_id))

        cases = {}
        for case_id in case_ids:
            #case_id = str(case_id)
            case_params = {}
            #PATIENT_ID	gender	ethnicity	race	year_of_diagnosis	
            # year_of_birth	overall_survival	vital_status	disease_specific_survival	primary_site
            case_params['case_id'] = case_id
            case_params['directory'] = directory.joinpath(str(case_id))
            case_params['case_metadata'] = case_metadatas[case_id]
            case_params['case_file_paths'] = cases_file_paths
            case_params['genomic_type'] = genomic_type
            #get from the clinical DF the row that corresponds to the case id == PATIENT_ID
            case_params['clinical_data'] = self.clinical_data_df.loc[case_id]

            case_params['primary_site'] = case_params['clinical_data']['primary_site']
            case_params['vital_status'] = case_params['clinical_data']['vital_status']
            case_params['overall_survival'] = case_params['clinical_data']['overall_survival']
            case_params['disease_specific_survival'] = case_params['clinical_data']['disease_specific_survival']
            case_params['gender'] = case_params['clinical_data']['gender']
            case_params['ethnicity'] = case_params['clinical_data']['ethnicity']
            case_params['race'] = case_params['clinical_data']['race']


            

            cases[case_id] = case_params#TCGA_Case(**case_params)

        return cases

    def _get_genomic_data_wrapper(self, t_case):
        '''
        The multi-threaded wrapper of getting genomic data from a case.

        :param t_case: The dictionary pair of (case_id, Case).
        Case is also a dictionary now 
        '''
        case_id, case = t_case        
        #print("Genetic df", self.genetic_data_df.columns)
        # choose the row in genetic data that corresponds to the case id in the patient id column
        
        genomic_data = self.genetic_data_df[case_id]
        return case_id, genomic_data

    def _concat_genomic_data(self, cases, cache_directory, n_threads):
        '''
        Concatenate the genomic data from the cases.

        :param cases: The TCGA Case instances.
        :param cache_directory: Specify the directory for the cache files.
        :param n_threads: The number of threads to user for concatenating genomic data.
        '''
        # Check if the cache data exists
        genomic_latest_file_path = check_cache_files(cache_directory, regex=f'genomic_{self.genomic_type}_*')

        if genomic_latest_file_path:
            genomic_latest_file_created_date = genomic_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using genomic {} cache files created at {} for {}'.format(
                self.genomic_type,
                datetime.strptime(genomic_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))
            

            df_genomic_cache = pd.read_csv(genomic_latest_file_path, sep='\t', index_col='gene_id')

            return df_genomic_cache

        self.logger.info(f"Concatenating {len(cases)} cases' genomic {self.genomic_type} data for {self.project_id}...")
        df_genomic = self.genetic_data_df

        

        # Save the result to cache directory
        df_genomic.to_csv(
            cache_directory.joinpath(f'genomic_{self.genomic_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_genomic

    def _concat_clinical_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the clinical data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        clinical_latest_file_path = check_cache_files(cache_directory=cache_directory, regex='clinical_*')

        if clinical_latest_file_path:
            clinical_latest_file_created_date = clinical_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using clinical cache files created at {} for {}'.format(
                datetime.strptime(clinical_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_clinical_cache = pd.read_csv(clinical_latest_file_path, sep='\t', index_col='clinical')

            return df_clinical_cache.T

        self.logger.info('Concatenating {} cases\' clinical data for {}...'.format(len(cases), self.project_id))
        df_clinical = self.clinical_data_df

        
        # Save the clinical data
        df_clinical.to_csv(
            cache_directory.joinpath(f'clinical_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_clinical

    def _concat_vital_status_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the vital status data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        vital_status_latest_file_path = check_cache_files(cache_directory=cache_directory, regex='vital_status_*')

        if vital_status_latest_file_path:
            vital_status_latest_file_created_date = vital_status_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using vital status cache files created at {} for {}'.format(
                datetime.strptime(vital_status_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_vital_status_cache = pd.read_csv(vital_status_latest_file_path, sep='\t', index_col='vital_status')

            return df_vital_status_cache

        self.logger.info('Concatenating {} cases\' vital status data for {}...'.format(len(cases), self.project_id))
        # the first row is all the case ids, the second row is the value of the vital status


        df_vital_status = self.clinical_data_df['vital_status']

        # case_ids = self.case_ids
        # for case_id in case_ids:
        #     df_vital_status = df_vital_status.join(cases[case_id]['vital_status'], how='outer')

        # Add the name for index
        df_vital_status.index.rename(name='vital_status', inplace=True)

        # Save the vital status data
        df_vital_status.to_csv(
            cache_directory.joinpath(f'vital_status_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_vital_status

    def _concat_overall_survival_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the 5 years overall survival data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        overall_survival_latest_file_path = check_cache_files(cache_directory, regex=r'overall_survival_*')

        if overall_survival_latest_file_path:
            latest_file_created_date = overall_survival_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using overall survival cache files created at {} for {}'.format(
                datetime.strptime(latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_overall_survival_cache = pd.read_csv(overall_survival_latest_file_path, sep='\t',
                                                    index_col='overall_survival')

            return df_overall_survival_cache

        self.logger.info('Concatenating {} cases\' overall survival data for {}...'.format(len(cases), self.project_id))
        df_overall_survival = self.clinical_data_df['overall_survival']

        # case_ids = self.case_ids
        # for case_id in case_ids:
        #     df_overall_survival = df_overall_survival.join(cases[case_id]['overall_survival'], how='outer')

        # Add the name for index
        df_overall_survival.index.rename(name='overall_survival', inplace=True)

        # Save the overall survival data
        df_overall_survival.to_csv(
            cache_directory.joinpath(f'overall_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_overall_survival

    def _concat_disease_specific_survival_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the 5 years disease specific survival data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'disease_specific_survival_*')

        if latest_file_path:
            disease_specific_survival_latest_file_created_date = latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using disease specific survival cache files created at {} for {}'.format(
                datetime.strptime(disease_specific_survival_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_disease_specific_survival_cache = pd.read_csv(latest_file_path, sep='\t',
                                                             index_col='disease_specific_survival')

            return df_disease_specific_survival_cache

        self.logger.info(f'Concatenating {len(cases)} cases\' disease specific survival data for {self.project_id}...')
        df_disease_specific_survival = pd.DataFrame()

        
        df_disease_specific_survival = self.clinical_data_df['disease_specific_survival']

        # Add the name for index
        df_disease_specific_survival.index.rename(name='disease_specific_survival', inplace=True)

        # Save the disease specific survival data
        df_disease_specific_survival.to_csv(
            cache_directory.joinpath(f'disease_specific_survival_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_disease_specific_survival

    def _concat_survival_time_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the survival time data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        survival_time_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'survival_time*')

        if survival_time_latest_file_path:
            survival_time_latest_file_created_date = survival_time_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using survival time cache files created at {} for {}'.format(
                datetime.strptime(survival_time_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_survival_time_cache = pd.read_csv(survival_time_latest_file_path, sep='\t', index_col='survival_time')

            return df_survival_time_cache

        self.logger.info('Concatenating {} cases\' survival time data for {}...'.format(len(cases), self.project_id))
        df_survival_time = self.clinical_data_df['overall_survival']

        # case_ids = genomic_data.columns.to_list()
        # for case_id in case_ids:
        #     df_survival_time = df_survival_time.join(cases[case_id].survival_time, how='outer')

        # Add the name for index
        df_survival_time.index.rename(name='survival_time', inplace=True)

        # Save the survival time data
        df_survival_time.to_csv(
            cache_directory.joinpath(f'survival_time_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_survival_time

    def _concat_primary_site_data(self, cases, genomic_data, cache_directory):
        '''
        Concatenate the primary site data from the cases.

        :param cases: The TCGA Case instances.
        :param genomic_data: The genomic data used for aligning the case id (Deprecated).
        :param cache_directory: Specify the directory for the cache files.
        '''
        # Check if the cache data exists
        primary_site_latest_file_path = check_cache_files(cache_directory=cache_directory, regex=r'primary_site_*')

        if primary_site_latest_file_path:
            primary_site_latest_file_created_date = primary_site_latest_file_path.name.split('.')[0].split('_')[-1]
            self.logger.info('Using primary site cache files created at {} for {}'.format(
                datetime.strptime(primary_site_latest_file_created_date, "%Y%m%d%H%M%S"),
                self.project_id
            ))

            df_primary_site_cache = pd.read_csv(primary_site_latest_file_path, sep='\t',
                                                index_col='primary_site', dtype='category')

            return df_primary_site_cache

        self.logger.info('Concatenating {} cases\' primary site data for {}...'.format(len(cases), self.project_id))
        df_primary_site = self.clinical_data_df['primary_site']

        

        # Add the name for index
        df_primary_site.index.rename(name='primary_site', inplace=True)

        # Save the primary site data
        df_primary_site.to_csv(
            cache_directory.joinpath(f'primary_site_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'),
            sep='\t'
        )
        self.logger.info('Saving concatenate results for {} to cache file'.format(self.project_id))

        return df_primary_site


            

# !Need to modify in the future
    def _get_chosen_gene_ids(self, cache_directory, well_known_gene_ids):

        chosen_gene_ids = well_known_gene_ids

        self.logger.info(f'Chosen gene ids: {" ".join(chosen_gene_ids)}')


    @property
    def genomic(self):
        '''
        Return the genomic data with gene_ids.
        '''
        return self._genomic

    @property
    def clinical(self):
        '''
        Return the clinical data.
        '''
        return self._clinical

    @property
    def vital_status(self):
        '''
        Return the vital status data.
        '''
        return self._vital_status

    @property
    def overall_survival(self):
        '''
        Return the 5 year overall survival data.
        '''
        return self._overall_survival

    @property
    def disease_specific_survival(self):
        '''
        Return the 5 year disease specific survival data.
        '''
        return self._disease_specific_survival

    @property
    def survival_time(self):
        '''
        Return the survival time data.
        '''
        return self._survival_time

    @property
    def primary_site(self):
        '''
        Return the primary site data.
        '''
        return self._primary_site

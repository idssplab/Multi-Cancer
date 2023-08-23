from dataset import TCGA_Project_Dataset, TCGA_Program_Dataset, METABRIC_Dataset
from utils.logger import setup_logging
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests/'))

    project_id = 'TCGA-BRCA'
    download_root_directory = './Data'
    cache_root_directory = './Cache'
    n_threads = 16

    chosen_features = {
        #'gene_ids': 'ALL',
        'gene_ids': {
            'TCGA-BRCA': ['ESR1', 'MYC', 'HEXIM1', 'HNRNPU', 'ACTB', 'BTF3', 'PINK1', 'RPS9', 'RPL36', 'RPL18', 'RPL28', 'PAN2', 'SRSF5', 'OCIAD1', 'TCTN2', 'YWHAH', 'MKI67', 'ERBB2', 'PGR', 'PLAU'],
            'TCGA-LUAD': ['TUBA4A', 'RAB4A', 'COX4I1', 'NINL', 'SRSF6', 'IFI16', 'ZYX', 'EFTUD2', 'NXF1', 'EPB41L5', 'TNIP2', 'BIRC3', 'CEP70', 'HIF1A', 'PKM', 'SLC2A1', 'ALCAM', 'CADM1', 'EPCAM', 'PTK7'],
            'TCGA-COAD': ['ZBTB2', 'PARP1', 'VAPA', 'NDUFS8', 'XPO1', 'DDX39A', 'CIT', 'AGR2', 'HNRNPA1', 'GSK3B', 'MEPCE', 'CDK2', 'CD44', 'ABCC1', 'ALDH1A1', 'ABCB1', 'ABCG2', 'ALCAM', 'EPCAM', 'PROM1']
        },
        'clinical_numerical_ids': [
            'age_at_diagnosis',
            'year_of_diagnosis',
            'year_of_birth'
        ],
        'clinical_categorical_ids': [
            'gender',
            'race',
            'ethnicity',
        ]
    }

    #METABRIC_Dataset = METABRIC_Dataset(
    #    data_directory='/'.join([download_root_directory, 'METABRIC']),
    #    cache_directory='/'.join([cache_root_directory, 'METABRIC'])
    #)

    #Project_Dataset = TCGA_Project_Dataset(
    #    project_id=project_id,
    #    chosen_features=chosen_features,
    #    data_directory='/'.join([download_root_directory, project_id]),
    #    cache_directory='/'.join([cache_root_directory, project_id]),
    #    n_threads=n_threads
    #)

    Program_Dataset = TCGA_Program_Dataset(
        project_ids=['TCGA-BRCA', 'TCGA-LUAD', 'TCGA-COAD'],
        chosen_features=chosen_features,
        data_directory=download_root_directory,
        cache_directory=cache_root_directory,
        n_threads=n_threads
    )

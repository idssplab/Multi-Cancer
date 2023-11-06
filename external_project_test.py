from utils.api import get_filters_result_from_project
from utils.logger import setup_logging
from utils.util import set_random_seed
from pathlib import Path
from preprocess.external_project import External_Project 

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests/'))

    project_id = 'SCLC'
    download_root_directory = './Data/sclc_ucologne_2015'
    cache_root_directory = './Cache'
    n_threads = 16

    Project = External_Project(project_id=project_id,
        genomic_type='tpm',
        well_known_gene_ids=['TP53', 'RB1', 'CREBBP', 'EP300', 'TP73', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'RBL1', 'RBL2', 'PTEN', 'MYCL1', 'MYCN', 'MYC', 'FGFR1', 'IRS2', 'BRAF', 'KIT', 'PIK3CA'],
        download_directory='/'.join([download_root_directory, project_id]),
        cache_directory='/'.join([cache_root_directory, project_id]),
        n_threads=n_threads
    )

    project_filters = {
        '=': {'program.name': 'external'}
    }
    project_ids = [project_metadata['id'] for project_metadata in get_filters_result_from_project(filters=project_filters, sort='summary.case_count:desc', size=100)]

    for project_id in project_ids:
        TCGA_Project(project_id=project_id,
            genomic_type='tpm',
            download_directory='/'.join([download_root_directory, project_id]),
            cache_directory='/'.join([cache_root_directory, project_id]),
            n_threads=n_threads
        )

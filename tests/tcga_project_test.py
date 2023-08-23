from utils.api import get_filters_result_from_project
from preprocess import TCGA_Project
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

    Project = TCGA_Project(project_id=project_id,
        genomic_type='tpm',
        #well_known_gene_ids=['ESR1', 'PGR', 'ERBB2', 'MKI67', 'PLAU'],
        download_directory='/'.join([download_root_directory, project_id]),
        cache_directory='/'.join([cache_root_directory, project_id]),
        n_threads=n_threads
    )

    project_filters = {
        '=': {'program.name': 'TCGA'}
    }
    project_ids = [project_metadata['id'] for project_metadata in get_filters_result_from_project(filters=project_filters, sort='summary.case_count:desc', size=100)]

    for project_id in project_ids:
        TCGA_Project(project_id=project_id,
            genomic_type='tpm',
            download_directory='/'.join([download_root_directory, project_id]),
            cache_directory='/'.join([cache_root_directory, project_id]),
            n_threads=n_threads
        )

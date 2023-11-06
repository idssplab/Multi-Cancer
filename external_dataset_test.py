from preprocess.external_dataset import ExternalDataset
from utils.logger import setup_logging
from pathlib import Path
from utils.util import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests/'))

    download_root_directory = './Data/sclc_ucologne_2015'
    cache_root_directory = './Cache/sclc_ucologne_2015'

    project = ExternalDataset(project_id='sclc_ucologne_2015',
        download_directory=download_root_directory,
        cache_directory=cache_root_directory)
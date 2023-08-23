from preprocess import METABRIC
from utils.logger import setup_logging
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests/'))

    download_root_directory = './Data/METABRIC'
    cache_root_directory = './Cache/METABRIC'

    Project = METABRIC(
        download_directory=download_root_directory,
        cache_directory=cache_root_directory
    )
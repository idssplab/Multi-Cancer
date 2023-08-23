import logging
from utils.logger import setup_logging
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests'))

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    print(loggers)

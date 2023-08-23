from utils.logger import setup_logging, get_logger, TensorboardWriter
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests'))
    tensorboard_logger = get_logger('Tensorboard')
    writer = TensorboardWriter(Path('./Logs/Tests'), tensorboard_logger, True)

    writer.set_step(0)
    writer.add_scalar('test', 0)

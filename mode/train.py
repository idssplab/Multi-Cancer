import dataset as module_dataset
import datasets_manager as module_datasets_manager
import runner as module_runner
from utils.logger import get_logger

def multi_train_bootstrap(config):
    logger = get_logger('runner.multi_train_bootstrap')

    Datasets = {project_id: config.init_obj(
            f'datasets.{project_id}',
            module_dataset
        )
        for project_id in config['datasets']
    }

    Datasets_Manager = getattr(
        module_datasets_manager,
        config['datasets_manager.type']
    )(datasets=Datasets, config=config)

    project_results = {}
    for project_id in Datasets:
        logger.info('Training Start')
        runner_init_config = {
            'config': config,
            'train_data_loader': Datasets_Manager[project_id]['dataloaders']['train'],
            'test_data_loader': Datasets_Manager[project_id]['dataloaders']['test']
        }

        runner = getattr(
            module_runner,
            config['runner.type']
        )(**runner_init_config)
        result = runner.run()

        project_results[project_id] = result
        logger.info('Training End')

    return project_results

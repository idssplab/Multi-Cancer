import pandas as pd
import dataset as module_dataset
import datasets_manager as module_datasets_manager
import runner as module_runner
from utils.logger import get_logger


def multi_cross_validation(config):
    logger = get_logger('runner.multi_cross_validation')

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
        project_result = {}
        logger.info('Cross Validation Start')
        for fold_index in Datasets_Manager[project_id]['dataloaders']:
            logger.info('{} Fold for {}...'.format(fold_index+1, project_id))

            runner_init_config = {
                'config': config,
                'train_data_loader': Datasets_Manager[project_id]['dataloaders'][fold_index]['train'],
                'valid_data_loader': Datasets_Manager[project_id]['dataloaders'][fold_index]['valid']
            }

            runner = getattr(
                module_runner,
                config['runner.type']
            )(**runner_init_config)
            result = runner.run()

            project_result[fold_index] = {k: v['mean'] for k, v in result.items()}
        
        project_results[project_id] = pd.DataFrame.from_dict(project_result, 'index').describe().T.loc[:, ['mean', 'std']].to_dict('index')

        logger.info('Cross Validation End')

    return project_results

def multi_cross_validation_bootstrap(config):
    logger = get_logger('runner.multi_cross_validation_bootstrap')

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

    project_results = {
        'cross_validation': {},
        'bootstrap': {}
    }
    for project_id in Datasets:
        project_cross_validation_result = {}
        logger.info('Cross Validation Start')
        for fold_index in Datasets_Manager[project_id]['dataloaders']:
            if isinstance(fold_index, int):
                logger.info('{} Fold for {}...'.format(fold_index+1, project_id))

                runner_init_config = {
                    'config': config,
                    'train_data_loader': Datasets_Manager[project_id]['dataloaders'][fold_index]['train'],
                    'valid_data_loader': Datasets_Manager[project_id]['dataloaders'][fold_index]['valid']
                }

                runner = getattr(
                    module_runner,
                    config['runner.type']
                )(**runner_init_config)
                result = runner.run()

                project_cross_validation_result[fold_index] = {k: v['mean'] for k, v in result.items()}
        
        project_results['cross_validation'][project_id] = pd.DataFrame.from_dict(
            project_cross_validation_result,'index'
        ).describe().T.loc[:, ['mean', 'std']].to_dict('index')

        logger.info('Cross Validation End')
        logger.info('Bootstrapping Start')

        runner_init_config = {
            'config': config,
            'train_data_loader': Datasets_Manager[project_id]['dataloaders']['train'],
            'test_data_loader': Datasets_Manager[project_id]['dataloaders']['test']
        }

        runner = getattr(
            module_runner,
            config['runner.type']
        )(**runner_init_config)
        project_bootstrap_result = runner.run()

        project_results['bootstrap'][project_id] = project_bootstrap_result
        logger.info('Bootstrapping End')

    return project_results

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from base import BaseDatasetsManager
from utils.logger import get_logger
from .sampler import SubsetWeightedRandomSampler, SubsetSampler, BootstrapSubsetSampler


class TCGA_Datasets_Manager(BaseDatasetsManager):
    '''
    A TCGA Datasets Manager
    '''
    def __init__(self, config, datasets):
        '''
        Initialize the TCGA Datasets Manager instance with parameters.

        Needed parameters
        :param datasets: The datasets managed by this datasets manager.
        :param config: The configuration dictionary.
        '''
        self.base_datasets_manager_init_kwargs = {
            'config': config,
            'datasets': datasets
        }

        self.logger = get_logger('preprocess.tcga_datasets_manager')
        self.logger.info('Initializing a TCGA Datasets Manager containing {} Datasets...'.format(len(datasets)))

        super().__init__(**self.base_datasets_manager_init_kwargs)

    @staticmethod
    def get_samplers(dataset, test_split):
        '''
        Get samplers.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_indices, test_indices = train_test_split(total_indices, test_size=test_split, stratify=dataset.stratified_targets)
            
            indices = {
                'train': train_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        train_sampler_method = SubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        samplers = {
            'train': train_sampler_method(train_indices),
            'test': test_sampler_method(test_indices, replacement=True)
        }

        return samplers

    @staticmethod
    def get_kfold_samplers(dataset, num_folds):
        '''
        Get KFold samplers.
        '''
        kfold_method = StratifiedKFold
        train_sampler_method = SubsetSampler
        valid_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds)

        kfold_samplers = {}
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data, y=dataset.targets)):
            split_samplers = {
                'train': train_sampler_method(train_indices),
                'valid': valid_sampler_method(valid_indices, replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

    @staticmethod
    def get_kfold_test_samplers(dataset, test_split, num_folds):
        '''
        Get KFold samplers, and a test sampler.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_valid_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_valid_indices, test_indices = train_test_split(total_indices, test_size=test_split, stratify=dataset.stratified_targets)
            
            indices = {
                'train': train_valid_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        kfold_method = StratifiedKFold
        train_sampler_method = SubsetSampler
        valid_sampler_method = BootstrapSubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds)

        kfold_samplers = {
            'train': train_sampler_method(train_valid_indices),
            'test': test_sampler_method(test_indices, replacement=True)
        }
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data[train_valid_indices], y=dataset.targets[train_valid_indices])):
            split_samplers = {
                'train': train_sampler_method(train_valid_indices[train_indices]),
                'valid': valid_sampler_method(train_valid_indices[valid_indices], replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

class TCGA_Balanced_Datasets_Manager(BaseDatasetsManager):
    '''
    A TCGA Balanced Datasets Manager
    '''
    def __init__(self, config, datasets):
        '''
        Initialize the TCGA Balanced Datasets Manager instance with parameters.

        Needed parameters
        :param datasets: The datasets managed by this datasets manager.
        :param config: The configuration dictionary.
        '''
        self.base_datasets_manager_init_kwargs = {
            'config': config,
            'datasets': datasets
        }

        self.logger = get_logger('preprocess.tcga_balanced_datasets_manager')
        self.logger.info('Initializing a TCGA Balanced Datasets Manager containing {} Datasets...'.format(len(datasets)))

        super().__init__(**self.base_datasets_manager_init_kwargs)

    @staticmethod
    def get_samplers(dataset, test_split):
        '''
        Get samplers.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_indices, test_indices = train_test_split(total_indices, test_size=test_split, stratify=dataset.stratified_targets)
            
            indices = {
                'train': train_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        train_sampler_method = SubsetWeightedRandomSampler
        test_sampler_method = BootstrapSubsetSampler

        train_weights = dataset.weights[train_indices]

        samplers = {
            'train': train_sampler_method(train_indices, train_weights),
            'test': test_sampler_method(test_indices, replacement=True)
        }

        return samplers

    @staticmethod
    def get_kfold_samplers(dataset, num_folds):
        '''
        Get KFold samplers.
        '''
        kfold_method = StratifiedKFold
        train_sampler_method = SubsetWeightedRandomSampler
        valid_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds, shuffle=True)

        kfold_samplers = {}
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data, y=dataset.targets)):
            train_weights = dataset.weights[train_indices]

            split_samplers = {
                'train': train_sampler_method(train_indices, train_weights),
                'valid': valid_sampler_method(valid_indices, replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

    @staticmethod
    def get_kfold_test_samplers(dataset, test_split, num_folds):
        '''
        Get KFold samplers, and a test sampler.
        '''
        indices = dataset.load_indices()

        if indices:
            if len(indices['train']) + len(indices['test']) == len(dataset):
                train_valid_indices, test_indices = indices['train'], indices['test']
            else:
                raise ValueError('Wrong indices file')
        else:
            total_indices = np.arange(len(dataset))
            train_valid_indices, test_indices = train_test_split(total_indices, test_size=test_split, stratify=dataset.stratified_targets)
            
            indices = {
                'train': train_valid_indices,
                'test': test_indices
            }
            dataset.save_indices(indices)

        kfold_method = StratifiedKFold
        train_sampler_method = SubsetWeightedRandomSampler
        valid_sampler_method = BootstrapSubsetSampler
        test_sampler_method = BootstrapSubsetSampler

        KFold = kfold_method(n_splits=num_folds, shuffle=True)

        train_weights = dataset.weights[train_valid_indices]

        kfold_samplers = {
            'train': train_sampler_method(train_valid_indices, train_weights),
            'test': test_sampler_method(test_indices, replacement=True)
        }
        for fold, (train_indices, valid_indices) in enumerate(KFold.split(X=dataset.data[train_valid_indices], y=dataset.targets[train_valid_indices])):
            train_weights = dataset.weights[train_valid_indices[train_indices]]

            split_samplers = {
                'train': train_sampler_method(train_valid_indices[train_indices], train_weights),
                'valid': valid_sampler_method(train_valid_indices[valid_indices], replacement=False),
            }

            kfold_samplers[fold] = split_samplers

        return kfold_samplers

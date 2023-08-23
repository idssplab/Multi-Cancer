import torch
from abc import abstractmethod
import numpy as np
from numpy import inf
from utils.logger import get_logger, TensorboardWriter
import model as module_model
import utils.runner.loss as module_loss
import utils.runner.metric as module_metric
import utils.runner.plot as module_plot


class BaseTrainer(object):
    '''
    Base class for all trainers
    '''
    def __init__(self, config):
        '''
        Initialize the Base Trainer instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        self.config = config
        self.logger = get_logger('runner.base_trainer')

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.non_blocking = config['pin_memory']
        self.models = {model_name :self.config.init_obj(f'models.{model_name}', module_model).to(self.device) for model_name in self.config['models']}
        if len(device_ids) > 1:
            for model_name in self.models:
                self.models[model_name] = torch.nn.parallel.DataParallel(self.models[model_name], device_ids=device_ids)

        # Setup optimizers
        self.optimizers = {}
        for optimizer_name in self.config['optimizers']:
            params = []
            for model_name in self.config[f'optimizers.{optimizer_name}.models']:
                params += list(self.models[model_name].parameters())
            self.optimizers[optimizer_name] = self.config.init_obj(f'optimizers.{optimizer_name}', torch.optim, params)
        
        # Setup learning rate schedulers
        if self.config['lr_schedulers'] is not None:
            self.lr_schedulers = {}
            for lr_scheduler_name in self.config['lr_schedulers']:
                optimizer_name = self.config[f'lr_schedulers.{lr_scheduler_name}.optimizer']
                self.lr_schedulers[lr_scheduler_name] = self.config.init_obj(f'lr_schedulers.{lr_scheduler_name}', torch.optim.lr_scheduler, self.optimizers[optimizer_name])
        else:
            self.lr_schedulers = None

        # Loss functions
        self.losses = {loss: config.init_obj(f'losses.{loss}', module_loss) for loss in config['losses']}

        # Metric functions
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Plot functions
        self.plots = [getattr(module_plot, plt) for plt in config['plots']]

        # Trainer parameters settings
        self.epochs = config['runner.epochs']
        self.save_epoch = config['runner'].get('save_epoch', self.epochs)
        self.log_epoch = config['runner'].get('log_epoch', 1)
        self.start_epoch = 1
        self.checkpoint_dir = config.ckpt_dir

        # Configuration to monitor model performance and save best
        self.monitor = config['runner'].get('monitor', 'off')
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            self.early_stop = config['runner'].get('early_stop', inf)

        # Setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, config['runner.tensorboard'])

        # Resume from the checkpoint
        if config.resume is not None:
            self._load_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch

        :param epoch: Current epoch number
        '''
        raise NotImplementedError

    def run(self):
        '''
        Full training logic
        '''
        results = {}

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Run training for one epoch
            result = self._train_epoch(epoch)

            # Save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # Evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    # Check whether model performance improved or not, according to specified metric(monitor_metric)
                    improved = (self.monitor_mode == 'min' and log[self.monitor_metric]['mean'] <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and log[self.monitor_metric]['mean'] >= self.monitor_best)
                except KeyError:
                    self.logger.warning('Metric "{}" is not found. Model performance monitoring is disabled.'.format(self.monitor_metric))
                    self.monitor_mode = 'off'
                    improved = False

                if improved:
                    self.monitor_best = log[self.monitor_metric]['mean']
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info('Validation performance didn\'t improve for {} epochs. Training stops.'.format(self.early_stop))
                    break

            # Update results
            if self.monitor_mode == 'off':
                results.update(result)
            else:
                if best:
                    results.update(result)
        
            # Log informations
            if epoch % self.log_epoch == 0 or best:
                for key, value in log.items():
                    if isinstance(value, dict):
                        self.logger.info('{:20s}: {:.5f} ±{:.5f}'.format(str(key).lower(), value['mean'], value['std']))
                    else:
                        self.logger.info('{:20s}: {}'.format(str(key).lower(), value))

            # Save model
            if epoch % self.save_epoch == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

        return results

    def _prepare_device(self, n_gpu_use):
        '''
        Setup GPU device if available, move model into configured device
        '''
        # Get the GPU counts
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning('There\'s no GPU available on this machine, training will be performed on CPU.')
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning('The number of GPU\'s configured to use is {}, but only {} are available on this machine.'.format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        # Get the device
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        '''
        Save checkpoint

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        '''
        state = {
            'epoch': epoch,
            'models': {model_name: model.state_dict() for model_name, model in self.models.items()},
            'optimizers': {optimizer_name: optimizer.state_dict() for optimizer_name, optimizer in self.optimizers.items()},
            'monitor_best': self.monitor_best,
            # 'config': self.config
        }

        if self.lr_schedulers is not None:
            state.update({'lr_schedulers': {lr_scheduler_name: lr_scheduler.state_dict() for lr_scheduler_name, lr_scheduler in self.lr_schedulers.items()}})

        normal_path = self.checkpoint_dir.joinpath(f'checkpoint-epoch{epoch}.pth')
        torch.save(state, normal_path)
        self.logger.info('Saving checkpoint: {}...'.format(normal_path))

        if save_best:
            best_path = self.checkpoint_dir.joinpath('checkpoint-best.pth')
            torch.save(state, best_path)
            self.logger.info('Saving current best checkpoint: {}...'.format(best_path))

    def _load_checkpoint(self, load_path, resume=True, model_names='ALL'):
        '''
        Load checkpoint

        :param load_path: Checkpoint path to be loaded
        :param resume: Decide if this checkpoint used for resume the training
        '''
        self.logger.info('Loading checkpoint: {}...'.format(load_path))

        checkpoint = torch.load(load_path)
        self.logger.info('Checkpoint loaded')

        if model_names == 'ALL':
            for model_name in self.models:
                self.models[model_name].load_state_dict(checkpoint['models'][model_name])
        else:
            for model_name in model_names:
                self.models[model_name].load_state_dict(checkpoint['models'][model_name])
        self.logger.info('Loading {} models'.format(model_names))
        
        if resume:
            self.start_epoch = checkpoint['epoch'] + 1
            self.monitor_best = checkpoint['monitor_best']

            for optimizer_name in self.optimizers:
                self.optimizers[optimizer_name].load_state_dict(checkpoint['optimizers'][optimizer_name])

            if self.lr_schedulers is not None:
                for lr_scheduler_name in self.lr_schedulers:
                    self.lr_schedulers[lr_scheduler_name].load_state_dict(checkpoint['lr_schedulers'][lr_scheduler_name])

            self.logger.info('Resume training from epoch {}'.format(self.start_epoch))
    
    def _save_bootstrap_status(self, project_ids, bootstrap_status):
        '''
        Save bootstrap statuses

        :param bootstrap_statuses: 
        '''
        for k, v in bootstrap_status.items():
            bootstrap_status[k] = np.stack(v, axis=0)

        project_ids = '_'.join(project_ids)
        project_ids = project_ids.lower()

        np.savez(self.config.log_dir.joinpath(f'{project_ids}_bootstrap_status.npz'), **bootstrap_status)
        self.logger.info('Saving bootstrap status to {}'.format(self.config.log_dir))

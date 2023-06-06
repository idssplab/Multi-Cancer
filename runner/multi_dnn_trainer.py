import torch
from base import BaseTrainer
from .tracker import MetricTracker


class Multi_DNN_Trainer(BaseTrainer):
    '''
    Multi DNN Trainer
    '''
    def __init__(self, config, train_data_loader, valid_data_loader=None, test_data_loader=None):
        '''
        Initialize the Multi DNN Trainer instance with parameters.

        Needed parameters
        :param config: The configuration dictionary.
        '''
        super().__init__(config)

        # Dataloaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        # Trainer parameters settings
        self.len_epoch = len(self.train_data_loader)
        self.bootstrap_project_id_indices = config['runner'].get('bootstrap_project_id_indices', [''])

        # Metric trackers
        self.train_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss'], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            epoch_keys=[m.__name__ for m in self.metrics],
            iter_keys=['loss'], writer=self.writer
        )
        self.test_metrics = MetricTracker(
            iter_keys=[m.__name__ if i == '' else f'{i}_{m.__name__}' for m in self.metrics for i in self.bootstrap_project_id_indices],
            epoch_keys=[], writer=self.writer
        )

    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch

        :param epoch: Current epoch number
        '''
        # Set models to training mode
        for model_name in self.models:
            self.models[model_name].train()

        # Reset train metric tracker
        self.train_metrics.reset()

        # Start training
        outputs = []
        targets = []
        survival_times = []
        vital_statuses = []
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            genomic, clinical, _, project_id = data
            target, survival_time, vital_status = target

            # Transfer device and dtype
            genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
            target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            survival_time = survival_time.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
            vital_status = vital_status.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

            # Extract Features
            if self.models.get('Feature_Extractor', None) is not None:
                feature_extractor_name = 'Feature_Extractor'
                embedding = self.models['Feature_Extractor'](genomic, clinical)
            elif self.models.get('Genomic_Separate_Feature_Extractor', None) is not None:
                feature_extractor_name = 'Genomic_Separate_Feature_Extractor'
                embedding = self.models['Genomic_Separate_Feature_Extractor'](genomic, clinical, project_id)
            else:
                raise KeyError('Please specify a valid feature extractor')

            # Label Classifier
            if self.models.get('Label_Classifier', None) is not None:
                classifier_name = 'Label_Classifier'
                output = self.models['Label_Classifier'](embedding)
            elif self.models.get('Task_Classifier', None) is not None:
                classifier_name = 'Task_Classifier'
                output = self.models['Task_Classifier'](embedding, project_id)
            else:
                raise KeyError('Please specify a valid classifier')

            # Train models for one iteration
            self.optimizers[feature_extractor_name].zero_grad()
            self.optimizers[classifier_name].zero_grad()
            loss = self.losses['bce_with_logits_loss'](output, target)
            loss.backward()
            self.optimizers[feature_extractor_name].step()
            self.optimizers[classifier_name].step()

            # Append output and target
            outputs.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            survival_times.append(survival_time.detach().cpu())
            vital_statuses.append(vital_status.detach().cpu())

            # Update train metric tracker for one iteration
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode='train')
            self.train_metrics.iter_update('loss', loss.item())

        # Concatenate the outputs and targets
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        survival_times = torch.cat(survival_times)
        vital_statuses = torch.cat(vital_statuses)
        
        # Update train metric tracker for one epoch
        for metric in self.metrics:
            if metric.__name__ == 'c_index':
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
            else:
                self.train_metrics.epoch_update(metric.__name__, metric(outputs, targets))

        # Update log for one epoch
        log = {'train_'+k: v for k, v in self.train_metrics.result().items()}

        # Validation
        if self.valid_data_loader:
            valid_log = self._valid_epoch(epoch)
            log.update(**{'valid_'+k: v for k, v in valid_log.items()})

        # Update learning rate if there is lr scheduler
        if self.lr_schedulers is not None:
            for lr_scheduler_name in self.lr_schedulers:
                self.lr_schedulers[lr_scheduler_name].step()

        # Testing
        if epoch == self.epochs:
            if self.test_data_loader:
                test_log = self._bootstrap()
                log.update(**{'bootstrap_'+k: v for k, v in test_log.items()})

        return log

    def _valid_epoch(self, epoch):
        '''
        Validating after training an epoch

        :param epoch: Current epoch number
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset valid metric tracker
        self.valid_metrics.reset()

        # Start validating
        outputs = []
        targets = []
        survival_times = []
        vital_statuses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                genomic, clinical, _, project_id = data
                target, survival_time, vital_status = target

                # Transfer device and dtype
                genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
                target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                survival_time = survival_time.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                vital_status = vital_status.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

                # Extract Features
                if self.models.get('Feature_Extractor', None) is not None:
                    embedding = self.models['Feature_Extractor'](genomic, clinical)
                elif self.models.get('Genomic_Separate_Feature_Extractor', None) is not None:
                    embedding = self.models['Genomic_Separate_Feature_Extractor'](genomic, clinical, project_id)
                else:
                    raise KeyError('Please specify a valid feature extractor')
                
                # Label Classifier
                if self.models.get('Label_Classifier', None) is not None:
                    output = self.models['Label_Classifier'](embedding)
                elif self.models.get('Task_Classifier', None) is not None:
                    output = self.models['Task_Classifier'](embedding, project_id)
                else:
                    raise KeyError('Please specify a valid classifier')

                # Loss for one batch
                loss = self.losses['bce_with_logits_loss'](output, target)

                # Append output and target
                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())
                survival_times.append(survival_time.detach().cpu())
                vital_statuses.append(vital_status.detach().cpu())

                # Update valid metric tracker for one iteration
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, mode='valid')
                self.valid_metrics.iter_update('loss', loss.item())

            # Concatenate the outputs and targets
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            survival_times = torch.cat(survival_times)
            vital_statuses = torch.cat(vital_statuses)

            # Update valid metric tracker for one epoch
            for metric in self.metrics:
                if metric.__name__ == 'c_index':
                    self.valid_metrics.epoch_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
                else:
                    self.valid_metrics.epoch_update(metric.__name__, metric(outputs, targets))

        return self.valid_metrics.result()

    def _bootstrap(self, repeat_times=1000):
        '''
        Testing logic for a model with bootstrapping.

        :param model_name: Current model name.
        '''
        # Set models to validating mode
        for model_name in self.models:
            self.models[model_name].eval()

        # Reset test metric tracker
        self.test_metrics.reset()

        # Start bootstrapping
        bootstrap_status = {
            'genomic': [],
            'clinical': [],
            'index': [],
            'project_id': [],
            'output': [],
            'target': [],
            'survival_time': [],
            'vital_status': []
        }
        for idx in range(repeat_times):
            genomics = []
            clinicals = []
            indices = []
            project_ids = []
            outputs = []
            targets = []
            survival_times = []
            vital_statuses = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_data_loader):
                    genomic, clinical, index, project_id = data
                    target, survival_time, vital_status = target

                    # Transfer device and dtype
                    genomic = genomic.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    clinical = clinical.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    project_id = project_id.to(self.device, dtype=torch.int64, non_blocking=self.non_blocking)
                    target = target.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    survival_time = survival_time.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                    vital_status = vital_status.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

                    # Extract Features
                    if self.models.get('Feature_Extractor', None) is not None:
                        embedding = self.models['Feature_Extractor'](genomic, clinical)
                    elif self.models.get('Genomic_Separate_Feature_Extractor', None) is not None:
                        embedding = self.models['Genomic_Separate_Feature_Extractor'](genomic, clinical, project_id)
                    else:
                        raise KeyError('Please specify a valid feature extractor')
                    
                    # Label Classifier
                    if self.models.get('Label_Classifier', None) is not None:
                        output = self.models['Label_Classifier'](embedding)
                    elif self.models.get('Task_Classifier', None) is not None:
                        output = self.models['Task_Classifier'](embedding, project_id)
                    else:
                        raise KeyError('Please specify a valid classifier')

                    # Append output and target
                    genomics.append(genomic.detach().cpu())
                    clinicals.append(clinical.detach().cpu())
                    indices.append(index.detach().cpu())
                    project_ids.append(project_id.detach().cpu())
                    outputs.append(output.detach().cpu())
                    targets.append(target.detach().cpu())
                    survival_times.append(survival_time.detach().cpu())
                    vital_statuses.append(vital_status.detach().cpu())

                # Concatenate the outputs and targets
                genomics = torch.cat(genomics)
                clinicals = torch.cat(clinicals)
                indices = torch.cat(indices)
                project_ids = torch.cat(project_ids)
                outputs = torch.cat(outputs)
                targets = torch.cat(targets)
                survival_times = torch.cat(survival_times)
                vital_statuses = torch.cat(vital_statuses)

                # Update test metric tracker for one epoch
                for metric in self.metrics:
                    for bootstrap_project_id_index in self.bootstrap_project_id_indices:
                        if metric.__name__ == 'c_index':
                            if bootstrap_project_id_index == '':
                                self.test_metrics.iter_update(metric.__name__, metric(outputs, survival_times, vital_statuses))
                            else:
                                self.test_metrics.iter_update(
                                    f'{bootstrap_project_id_index}_{metric.__name__}',
                                    metric(
                                        outputs[project_ids == bootstrap_project_id_index],
                                        survival_times[project_ids == bootstrap_project_id_index],
                                        vital_statuses[project_ids == bootstrap_project_id_index]
                                    )
                                )
                        else:
                            if bootstrap_project_id_index == '':
                                self.test_metrics.iter_update(metric.__name__, metric(outputs, targets))
                            else:
                                self.test_metrics.iter_update(
                                    f'{bootstrap_project_id_index}_{metric.__name__}',
                                    metric(
                                        outputs[project_ids == bootstrap_project_id_index],
                                        targets[project_ids == bootstrap_project_id_index]
                                    )
                                )
                # Record bootstrap status
                bootstrap_status['genomic'].append(genomics.numpy())
                bootstrap_status['clinical'].append(clinicals.numpy())
                bootstrap_status['index'].append(indices.numpy())
                bootstrap_status['project_id'].append(project_ids.numpy())
                bootstrap_status['output'].append(outputs.numpy())
                bootstrap_status['target'].append(targets.numpy())
                bootstrap_status['survival_time'].append(survival_times.numpy())
                bootstrap_status['vital_status'].append(vital_statuses.numpy())

        # Save bootstrap statuses
        self._save_bootstrap_status(self.test_data_loader.dataset.project_ids, bootstrap_status)

        return self.test_metrics.result()

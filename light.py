import argparse
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from sklearn import metrics
from tqdm import tqdm

from dataset import TCGA_Program_Dataset
from datasets_manager import TCGA_Datasets_Manager, TCGA_Balanced_Datasets_Manager
from model import Graph_And_Clinical_Feature_Extractor, Task_Classifier, Feature_Extractor
from utils import get_logger, set_random_seed, setup_logging

SEED = 1126
set_random_seed(SEED)


def main():
    # Select a config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    override_n_genes(config)                                                    # For multi-task graph models.
    config = config_add_subdict_key(sep='.', config=config)                     # For dataset_manager.
    config_name = Path(args.config).stem

    # Setup logging.
    setup_logging(log_path := f'Logs/{config_name}/{datetime.now():%Y-%m-%dT%H:%M:%S}/')
    logger = get_logger(config_name)
    logger.info(f'Using Random Seed {SEED} for this experiment')
    get_logger('lightning.pytorch.accelerators.cuda', log_level='WARNING')      # Disable cuda logging.
    filterwarnings('ignore', r'.*Skipping val loop.*')                          # Disable val loop warning.

    # Create dataset manager.
    data = {'TCGA_BLC': TCGA_Program_Dataset(**config['datasets'])}
    manager = TCGA_Datasets_Manager(datasets=data, config=config)

    # TODO: TCGA_Balanced_Datasets_Manager
    # TODO: Log hyperparameters.
    # TODO: lightning has it's own metrics.
    # TODO: Early stopping?

    # Cross validation.
    for key, values in manager['TCGA_BLC']['dataloaders'].items():
        if isinstance(key, int) and config['cross_validation']:
            models, optimizers = create_models_and_optimizers(config)
            model = LitFullModel(models, optimizers)
            trainer = pl.Trainer(
                default_root_dir=log_path,
                max_epochs=config['max_epochs'],
                log_every_n_steps=1,
                enable_model_summary=False,
                enable_checkpointing=False,
            )
            trainer.fit(model, train_dataloaders=values['train'], val_dataloaders=values['valid'])
        elif key == 'train':
            train = values
        elif key == 'test':
            test = values

    # Train the final model.
    models, optimizers = create_models_and_optimizers(config)
    model = LitFullModel(models, optimizers)
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=config['max_epochs'],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train)

    # Test the final model.
    bootstrap_results = []
    for _ in tqdm(range(config['bootstrap_repeats']), desc='Bootstrapping'):
        bootstrap_results.append(trainer.test(model, dataloaders=test, verbose=False)[0])
    bootstrap_results = pd.DataFrame.from_records(bootstrap_results)
    for key, value in bootstrap_results.describe().loc[['mean', 'std']].to_dict().items():
        logger.info(f'| {key.ljust(10).upper()} | {value["mean"]:.5f} ± {value["std"]:.5f} |')


def config_add_subdict_key(config: dict = None, prefix: str = '', sep: str = '.'):
    """Add the key of the sub-dict to the parent dict recursively with the separator."""
    if config is None:
        return None
    flatten_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flatten_dict.update(config_add_subdict_key(prefix=f'{prefix}{key}{sep}', sep=sep, config=value))
        flatten_dict[f'{prefix}{key}'] = value
    return flatten_dict


def override_n_genes(config: dict):
    all_listed_genes = config['datasets']['chosen_features']['gene_ids']
    if isinstance(all_listed_genes, list):
        n_genes = len(all_listed_genes)
    elif isinstance(all_listed_genes, dict):
        genes_set = set()
        for listed_genes in all_listed_genes.values():
            genes_set.update(listed_genes)
        n_genes = len(genes_set)
    else:
        raise ValueError(f'Unknown type of chosen_features: {type(all_listed_genes)}')
    for model_name in config['models'].keys():
        if 'n_genes' in config['models'][model_name]:
            config['models'][model_name]['n_genes'] = n_genes


def create_models_and_optimizers(config: dict):
    models: dict[str, torch.nn.Module] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}

    # Setup models. Do not use getattr() for better IDE support.
    for model_name, kargs in config['models'].items():
        if model_name == 'Graph_And_Clinical_Feature_Extractor':
            models['feat_ext'] = Graph_And_Clinical_Feature_Extractor(**kargs)
        elif model_name == 'Feature_Extractor':
            models['feat_ext'] = Feature_Extractor(**kargs)
        elif model_name == 'Task_Classifier':
            models['clf'] = Task_Classifier(**kargs)
        else:
            raise ValueError(f'Unknown model type: {model_name}')

    # Setup optimizers. If the key is 'all', the optimizer will be applied to all models.
    for key, optim_dict in config['optimizers'].items():
        opt_name = next(iter(optim_dict))
        if key == 'all':
            params = [param for model in models.values() for param in model.parameters()]
            optimizers[key] = getattr(torch.optim, opt_name)(params, **optim_dict[opt_name])
        else:
            optimizers[key] = getattr(torch.optim, opt_name)(models[key].parameters(), **optim_dict[opt_name])
    return models, optimizers


class LitFullModel(pl.LightningModule):
    def __init__(self, models: dict[str, torch.nn.Module], optimizers: dict[str, torch.optim.Optimizer]):
        super().__init__()
        self.feat_ext = models['feat_ext']
        self.classifier = models['clf']
        self.optimizers_dict = optimizers
        self.validation_step_outputs = []
        # Disable automatic optimization for manual backward if there are multiple optimizers.
        if 'all' not in self.optimizers_dict:
            self.automatic_optimization = False

    def configure_optimizers(self):
        if 'all' in self.optimizers_dict:
            return self.optimizers_dict['all']
        return [self.optimizers_dict['feat_ext'], self.optimizers_dict['clf']]

    def training_step(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch

        if isinstance(self.optimizers(), list):
            opt_feat_ext, opt_classifier = self.optimizers()
            opt_feat_ext.zero_grad()
            opt_classifier.zero_grad()

        embedding = self.feat_ext(genomic, clinical, project_id)
        y = self.classifier(embedding, project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)

        if isinstance(self.optimizers(), list):
            self.manual_backward(loss)
            opt_feat_ext.step()
            opt_classifier.step()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def _shared_eval(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)
        outputs, labels, project_id = y.detach().cpu(), overall_survival.detach().cpu(), project_id.detach().cpu()
        self.validation_step_outputs.append({'outputs': outputs, 'labels': labels, 'project_id': project_id})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log('loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        outputs = torch.cat([outputs['outputs'] for outputs in self.validation_step_outputs])
        labels = torch.cat([outputs['labels'] for outputs in self.validation_step_outputs])
        roc = metrics.roc_auc_score(labels, outputs)
        prc = metrics.average_precision_score(labels, outputs)
        self.log('roc', roc, on_epoch=True, prog_bar=True)
        self.log('prc', prc, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        outputs = torch.cat([outputs['outputs'] for outputs in self.validation_step_outputs])
        labels = torch.cat([outputs['labels'] for outputs in self.validation_step_outputs])
        project_id = torch.cat([outputs['project_id'] for outputs in self.validation_step_outputs])
        for i in torch.unique(project_id):
            mask = project_id == i
            roc = metrics.roc_auc_score(labels[mask], outputs[mask])
            prc = metrics.average_precision_score(labels[mask], outputs[mask])
            self.log(f'roc_{i}', roc, on_epoch=True)
            self.log(f'prc_{i}', prc, on_epoch=True)
        self.validation_step_outputs.clear()


if __name__ == '__main__':
    main()

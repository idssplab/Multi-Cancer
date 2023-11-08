import lightning.pytorch as pl
import torch
import torchmetrics
from lifelines.utils import concordance_index

class LitFullModel(pl.LightningModule):
    def __init__(self, models: dict[str, torch.nn.Module], optimizers: dict[str, torch.optim.Optimizer], config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.feat_ext = models['feat_ext']
        self.classifier = models['clf']
        self.optimizers_dict = optimizers
        self.step_outputs = []
        self.step_labels = []
        self.step_project_ids = []
        self.step_survival_time = []
        self.step_vital_status = []
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
            self.optimizers()[0].zero_grad()
            self.optimizers()[1].zero_grad()

        embedding = self.feat_ext(genomic, clinical, project_id)
        y = self.classifier(embedding, project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)

        if isinstance(self.optimizers(), list):
            self.manual_backward(loss)
            self.optimizers()[0].step()
            self.optimizers()[1].step()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def _shared_eval(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)
        self.step_survival_time.append(survival_time.detach().cpu())
        self.step_vital_status.append(vital_status.detach().cpu())
        self.step_outputs.append(y.detach().cpu())
        self.step_labels.append(overall_survival.detach().cpu().type(torch.int))
        self.step_project_ids.append(project_id.detach().cpu())
        self.step_results.append({
            'output': y.detach().cpu(),
            'label': overall_survival.detach().cpu().type(torch.int64),
            'survival_time': survival_time.detach().cpu(),
            'vital_status': vital_status.detach().cpu(),
            'project_id': project_id.detach().cpu(),
        })
        
        return loss

    def _shared_epoch_end(self) -> None:
        # here I'm logging the validation data
        outputs = torch.cat(self.step_outputs)
        labels = torch.cat(self.step_labels)
        project_id = torch.cat(self.step_project_ids)
        survival_time = torch.cat(self.step_survival_time)
        vital_status = torch.cat(self.step_vital_status)
        for i in torch.unique(project_id):
            mask = project_id == i


         

            roc = torchmetrics.functional.auroc(outputs[mask], labels[mask], 'binary')
            prc = torchmetrics.functional.average_precision(outputs[mask], labels[mask], 'binary')
            #precision = torchmetrics.functional.precision(outputs[mask], labels[mask], 'binary')
            #recall = torchmetrics.functional.recall(outputs[mask], labels[mask], 'binary')
            #precision, recall, thresholds = torchmetrics.functional.precision_recall_curve(outputs[mask], labels[mask], 'binary')
            cindex = concordance_index(survival_time[mask], outputs[mask], vital_status[mask])




            self.log(f'roc_{i}', roc, on_epoch=True, on_step=False)
            self.log(f'prc_{i}', prc, on_epoch=True, on_step=False)
            #self.log(f'precision_{i}', precision, on_epoch=True, on_step=False)
            #self.log(f'recall_{i}', recall, on_epoch=True, on_step=False)
            self.log(f'cindex_{i}', cindex, on_epoch=True, on_step=False)

        self.step_outputs.clear()
        self.step_labels.clear()
        self.step_project_ids.clear()
        self.step_survival_time.clear()
        self.step_vital_status.clear()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log('loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end()
import lightning.pytorch as pl
import torch
import torchmetrics

from utils.runner.metric import youden_j, c_index


class LitFullModel(pl.LightningModule):
    def __init__(self, models: dict[str, torch.nn.Module], optimizers: dict[str, torch.optim.Optimizer], config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.feat_ext = models['feat_ext']
        self.classifier = models['clf']
        self.optimizers_dict = optimizers
        self.step_results = []                                                  # Slow but clean.
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
        #external dataset failing here
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)
        self.step_results.append({
            'output': y.detach().cpu(),
            'label': overall_survival.detach().cpu().type(torch.int64),
            'survival_time': survival_time.detach().cpu(),
            'vital_status': vital_status.detach().cpu(),
            'project_id': project_id.detach().cpu(),
        })
        return loss

    def _shared_epoch_end(self) -> None:
        outputs = torch.cat([result['output'] for result in self.step_results])
        outputs = torch.functional.F.sigmoid(outputs)                           # AUC and PRC will not be affected.
        labels = torch.cat([result['label'] for result in self.step_results])
        #print("outputs", outputs)
        #print("labels", labels)

        survival_time = torch.cat([result['survival_time'] for result in self.step_results])
        vital_status = torch.cat([result['vital_status'] for result in self.step_results])
        project_id = torch.cat([result['project_id'] for result in self.step_results])
        thres = youden_j(outputs, labels).astype('float')
        for i in torch.unique(project_id):
            mask = project_id == i
            roc = torchmetrics.functional.auroc(outputs[mask], labels[mask], 'binary')
            prc = torchmetrics.functional.average_precision(outputs[mask], labels[mask], 'binary')
            # precision = torchmetrics.functional.precision(outputs[mask], labels[mask], 'binary', threshold=thres)
            # recall = torchmetrics.functional.recall(outputs[mask], labels[mask], 'binary', threshold=thres)
            
            cindex = c_index(outputs[mask], survival_time[mask], vital_status[mask])
            self.log(f'AUC_{i}', roc, on_epoch=True, on_step=False)
            self.log(f'PRC_{i}', prc, on_epoch=True, on_step=False)
            
            # self.log(f'Precision_{i}', precision, on_epoch=True, on_step=False)
            # self.log(f'Recall_{i}', recall, on_epoch=True, on_step=False)
            self.log(f'C-Index_{i}', cindex, on_epoch=True, on_step=False)
            print(f"AUROC {roc:.6f} AUPRC {prc:.6f} cindex {cindex:.6f}  thres {thres:.6f}", end='\n')
        self.step_results.clear()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log('loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        return y.detach().cpu().numpy()

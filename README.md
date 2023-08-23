# Exploiting Common Patterns in Diverse Cancer Types via Multi-Task Learning - ReadMe

### Cancer type codes:

For the config files that use three datasets, the following indices correspond to these datasets:

1. BRCA: 0
2. LUAD: 1
3. COAD: 2

## Reproducing results:

For reproducing the results for the single-task and multi-tasks models you need to run:

```jsx
main.py -c [CONFIG NAME FILE]
```

The config file names are the following:

| Config Filename | Task | Notes |
| --- | --- | --- |
| tcga_brca_coad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave LUAD out |
| tcga_brca_luad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave COAD out |
| tcga_dnn_trainer_cross_validation_4_bootstrap_clinical_overall.yaml | Single |  |
| tcga_dnn_trainer_cross_validation_4_bootstrap_test.yaml | Single |  |
| tcga_luad_coad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave BRCA out |
| tcga_multi_dnn_trainer_cross_validation_4_bootstrap_test.yaml | Multi | Original |
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Without task description |
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_unordered_tpm_clinical_overall.yaml | Multi | Without Ordered RNA-Seq Data |
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_unweighed_tpm_clinical_overall.yaml | Multi | Without Weighted Random Sampler |
| tcga_multi_dnn_trainer_unique_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Unique RNA-Seq Feature Extractor |

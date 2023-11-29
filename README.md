---
title: Bimodal Neural Network for Cancer Prognosis Prediction
author:
  - Bo-Run Wu
  - Tsung-Wei Lin
  - Zow Ormazabal
---



# Exploiting Common Patterns in Diverse Cancer Types via Multi-Task Learning 
 We developed a Python code base to interact with the REST API provided by the GDC Portal for data filtering and download. Due to Python packages' limited functionality compared to those in other languages like the R package TCGAbiolinks, we resorted to using this custom Python code base. We filtered out patients lacking survival status, time, complete RNA-Seq, and clinical data. We downloaded the data of qualified patients using GDC API version 33.1, released on May 31, 2022 

## System Requirements

This code was tested on Ubuntu 18.04.6, with 46.9 GiB RAM, Intel® Core™ i7-8700 CPU @ 3.20GHz × 12 processor, and NVIDIA GeForce RTX 2080/PCIe/SSE2 graphics card. We used Python 3.10.9 and Pytorch 1.13.1.

Besides the packages listed in `requirements.txt`, [DGL](https://www.dgl.ai/pages/start.html) is required.

## Installation guide

For running this code, you will need to create en environment in Anaconda using the following command:

```conda env create --file environment.yaml```

Make sure you are using the same library versions as in the `environment.yaml` file so that you avoid any errors regarding version incompatibilities. 
Installing all the dependencies might take between 10 and 30 minutes.

## Reproducing results:

For reproducing the results for the single-task and multi-tasks models you need to run:

```jsx
main.py -c [CONFIG NAME FILE]
```

Each configuration file will give you the results for each of the modes and ablation studies that we performed. The following table summarizes which table corresponds to which table in our study.



| Config Filename | Task | Notes | Table
| --- | --- | --- |---|
| tcga_brca_coad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave LUAD out |3|
| tcga_brca_luad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave COAD out |3|
| tcga_dnn_trainer_cross_validation_4_bootstrap_test.yaml | Single |  | 1,2|
| tcga_luad_coad_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Leave BRCA out |3|
| tcga_multi_dnn_trainer_cross_validation_4_bootstrap_test.yaml | Multi | Original |2,3|
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Without task description |3|
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_unordered_tpm_clinical_overall.yaml | Multi | Without Ordered RNA-Seq Data |3|
| tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_unweighed_tpm_clinical_overall.yaml | Multi | Without Weighted Random Sampler |3|
| tcga_multi_dnn_trainer_unique_task_cross_validation_4_bootstrap_tpm_clinical_overall.yaml | Multi | Unique RNA-Seq Feature Extractor |3|


### Pytorch Lighning Framework
After the reviewing process, we created a Pytorch Lighning version of our model. For running this code, you need to run:

```jsx
light.py -c config/light/[CONFIG NAME FILE]
```
for the TCGA-only experiments. And:

```jsx
light_external.py -c config/light/[CONFIG NAME FILE]
```
for the configs that include "external" or "SCLC" in their filename.

| Config Filename | Task | Notes | Table
| --- | --- | --- |---|
| STL_BRCA.yaml | Single |  |2,3|
| STL_LUAD.yaml | Single |  |2,3|
| STL_COAD.yaml | Single |  |2,3|
| STL_BRCA_external.yaml | Single | External validation |4|
| STL_LUAD_external.yaml | Single | External validation |4|
| STL_COAD_external.yaml | Single | External validation |4|
| MTL_TCGA.yaml | Multi | Only TCGA  |2,4|
| MTL_train_SCLC_test.yaml | Multi |  External validation  |4|





### Example

```jsx
main.py -c config/tcga_multi_dnn_trainer_task_cross_validation_4_bootstrap_unweighed_tpm_clinical_overall.yaml
```

Running the above results will yield the results for all three cancers in 'Without weighted random sampler' category in Table 3.


### Expected output

#### Single task
```[INFO]	2023-08-23 12:12:51,572 - Cross Validation Start
[INFO]	2023-08-23 12:12:51,572 - 1 Fold for TCGA-BRCA...
[INFO]	2023-08-23 12:12:53,221 - epoch               : 10
[INFO]	2023-08-23 12:12:53,221 - train_auroc         : 0.73994 ±0.00000
[INFO]	2023-08-23 12:12:53,221 - train_auprc         : 0.27325 ±0.00000
[INFO]	2023-08-23 12:12:53,221 - train_c_index       : 0.71927 ±0.00000
[INFO]	2023-08-23 12:12:53,221 - train_recall        : 0.78689 ±0.00000
[INFO]	2023-08-23 12:12:53,221 - train_precision     : 0.16327 ±0.00000
[INFO]	2023-08-23 12:12:53,221 - train_loss          : 0.33689 ±0.03093
[INFO]	2023-08-23 12:12:53,221 - valid_auroc         : 0.60477 ±0.00000
[INFO]	2023-08-23 12:12:53,222 - valid_auprc         : 0.18471 ±0.00000
[INFO]	2023-08-23 12:12:53,222 - valid_c_index       : 0.59909 ±0.00000
[INFO]	2023-08-23 12:12:53,222 - valid_recall        : 0.30000 ±0.00000
[INFO]	2023-08-23 12:12:53,222 - valid_precision     : 0.27273 ±0.00000
[INFO]	2023-08-23 12:12:53,222 - valid_loss          : 0.34016 ±0.00701
```

The model will first train on each cancer dataset **separately** and then calculate the bootstraped results separately as well. Once the bootstrap for one cancer ends, the model will train on the data for a different cancer and then proceed to calculate the bootstrapped results. 

```[INFO]	2023-08-23 12:16:59,479 - bootstrap_auroc     : 0.55437 ±0.07822
[INFO]	2023-08-23 12:16:59,480 - bootstrap_auprc     : 0.35305 ±0.09661
[INFO]	2023-08-23 12:16:59,480 - bootstrap_c_index   : 0.55428 ±0.07729
[INFO]	2023-08-23 12:16:59,480 - bootstrap_recall    : 0.38975 ±0.20531
[INFO]	2023-08-23 12:16:59,480 - bootstrap_precision : 0.35371 ±0.17675
```

Each run takes between 5 to 7 seven minutes.

#### Multi task
Training:
```[INFO]  Cross Validation Start
[INFO]  1 Fold for TCGA_BLC...
[INFO]  epoch               : 10
[INFO]  train_auroc         : 0.74027 ±0.00000
[INFO]  train_auprc         : 0.44451 ±0.00000
[INFO]  train_c_index       : 0.70998 ±0.00000
[INFO]  train_recall        : 0.86752 ±0.00000
[INFO]  train_precision     : 0.27395 ±0.00000
[INFO]  train_loss          : 0.42498 ±0.04557
[INFO]  valid_auroc         : 0.73718 ±0.00000
[INFO]  valid_auprc         : 0.47157 ±0.00000
[INFO]  valid_c_index       : 0.71856 ±0.00000
[INFO]  valid_recall        : 0.69863 ±0.00000
[INFO]  valid_precision     : 0.30000 ±0.00000
[INFO]  valid_loss          : 0.41974 ±0.04541
```
Bootstrapping:

```
[INFO]  bootstrap_0_auroc                       : 0.83943 ±0.04359
[INFO]  bootstrap_1_auroc                       : 0.64477 ±0.05999
[INFO]  bootstrap_2_auroc                       : 0.71172 ±0.07293
[INFO]  bootstrap_0_auprc                       : 0.34872 ±0.09029
[INFO]  bootstrap_1_auprc                       : 0.50874 ±0.08170
[INFO]  bootstrap_2_auprc                       : 0.49778 ±0.10215
[INFO]  bootstrap_0_c_index                     : 0.82355 ±0.04298
[INFO]  bootstrap_1_c_index                     : 0.58646 ±0.04947
[INFO]  bootstrap_2_c_index                     : 0.69602 ±0.06749
[INFO]  bootstrap_0_recall                      : 0.77102 ±0.10841
[INFO]  bootstrap_1_recall                      : 0.50554 ±0.12414
[INFO]  bootstrap_2_recall                      : 0.63841 ±0.13585
[INFO]  bootstrap_0_precision                   : 0.27854 ±0.09747
[INFO]  bootstrap_1_precision                   : 0.55147 ±0.10589
[INFO]  bootstrap_2_precision                   : 0.38913 ±0.11199
```
Where the indices next to `bootstrap_` correspond to each cancer.

#### Cancer type codes:

For the config files that use three datasets and the outputs of the Bootstrapped results, the following indices correspond to the datasets in this order:

1. BRCA: 0
2. LUAD: 1
3. COAD: 2



### Data

Make sure to include the data for all three cancers in a folder called `Data` with subfolders `Data/TCGA-BRCA`, `Data/TCGA-COAD`, `Data/TCGA-LUAD`. 

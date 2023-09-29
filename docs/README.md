
# Notes for Developers

## History

This project was originally developed by Bo-Run Wu, and later modified by Tsung-Wei Lin and Zow Ormazabal.

## Code Style

PEP 8 should be followed as much as possible. The legacy code, however, is not fully compliant with PEP 8. We will gradually refactor the code to make it compliant with PEP 8 except for the following rules:

- The line length should be less than 120 characters.
- For consistency, we will follow the legacy code style for class names, i.e., `Pascel_Case_With_Underscores`.



## Modifications

### Datasets

`TCGA_Project_Dataset` and `TCGA_Program_Dataset` now will give identical results, while the `project_id` will always be 0 for `TCGA_Project_Dataset`.

```python
# Results from __getitem__
((genomic, clinical, index, project_id), (target, survival_time, vital_statuse))
```

Both of them support graph mode, by setting `graph_dataset=True`. The graph data will be stored in `self._genomics`, which means overriding the non-graph data.

### Models

The `forward` functions of all feature extractors and classifiers should take the same arguments, i.e.,

```python
def forward(self, genomic, clinical, project_id):
    ...
```

### Runners

Most of the training logic is implemented in `BaseTrainer`. The testing logic is implemented individually in each runner. The `DNN_Trainer` works with `TCGA_Project_Dataset`, while the `Multi_DNN_Trainer` works with `TCGA_Program_Dataset`.
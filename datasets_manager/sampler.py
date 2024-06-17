import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
import math


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class SubsetWeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices, weights, replacement=True):
        self.indices = indices
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        for i in torch.multinomial(self.weights, self.num_samples, self.replacement):
            yield self.indices[i]

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.indices)

class BootstrapSubsetSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices, replacement=False):
        self.indices = indices
        self.replacement = replacement

    def __iter__(self):
        if self.replacement:
            for _ in range(self.num_samples // 16):
                for i in torch.randint(high=self.num_samples, size=(16,)).tolist():
                    yield self.indices[i]
            for i in torch.randint(high=self.num_samples, size=(self.num_samples % 16,)).tolist():
                yield self.indices[i]
        else:
            for i in torch.randperm(self.num_samples):
                yield self.indices[i]

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.indices)
def create_stratified_sampler(target): # target is a labels for the overall_survival
    class_counts = {}
    for item in target:
        class_counts[item] = class_counts.get(item, 0) + 1
    weight_per_class = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    weights = [weight_per_class[t] for t in target]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class BootstrapSubsetSamplerDM(Sampler):
    """Sampler that allows sampling with or without replacement."""
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else len(data_source)
        self.generator = generator

    def __iter__(self):
        if self.replacement:
            indices = torch.randint(len(self.data_source), (self.num_samples,), generator=self.generator).tolist()
        else:
            indices = torch.randperm(len(self.data_source), generator=self.generator).tolist()[:self.num_samples]
        for i in indices:
            yield i

    def __len__(self):
        return self.num_samples
    
def compute_weights_by_task_id(project_ids):
    '''
    Calculate the weights for each data point in the dataset based on project IDs to be used in the weighted sampler.
    Weights are calculated as the inverse square root of the count of each unique project ID.
    '''
    project_ids_tensor = torch.tensor(project_ids, dtype=torch.long)
    unique_ids, counts = torch.unique(project_ids_tensor, return_counts=True)
    weights = torch.zeros_like(project_ids_tensor, dtype=torch.float64)
    
    for uid, count in zip(unique_ids, counts):
        weights[project_ids_tensor == uid] = math.sqrt(1.0 / count)
    
    return weights
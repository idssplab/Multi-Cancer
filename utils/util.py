import json
import random
from collections import OrderedDict
from pathlib import Path

import dgl
import numpy
import torch
import yaml
from six import iteritems


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def read_yaml(fname):
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    yaml.add_constructor(_mapping_tag, dict_constructor)

    fname = Path(fname)

    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_yaml(content, fname):
    def dict_representer(dumper, data):
        return dumper.represent_dict(iteritems(data))

    yaml.add_representer(OrderedDict, dict_representer)

    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)


def set_random_seed(seed: int):
    # Set python seed
    random.seed(seed)

    # Set numpy seed
    numpy.random.seed(seed)

    # Set pytorch seed, disable benchmarking and avoiding nondeterministic algorithms
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    # Set DGL seed.
    dgl.seed(seed)


def check_cache_files(cache_directory, regex):
    '''
    Check if the cache file exists.

    :param cache_directory: Specify the directory for the cache files.
    :param regex: Specify the patterns for searching in cache_directory.
    '''
    cache_file_paths = [file_path for file_path in cache_directory.rglob(regex) if file_path.is_file()]

    latest_file_path = None
    for cache_file_path in cache_file_paths:
        cache_file_name = cache_file_path.name.split('.')[0]

        if not latest_file_path:
            latest_file_path = cache_file_path
        else:
            latest_file_name = latest_file_path.name.split('.')[0]

            if int(cache_file_name.split('_')[-1]) > int(latest_file_name.split('_')[-1]):
                latest_file_path = cache_file_path

    return latest_file_path

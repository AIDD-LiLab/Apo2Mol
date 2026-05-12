import torch
from torch.utils.data import Subset
import pickle
from tqdm import tqdm
from .mySubset import GetSubset
from .pl_pair_dataset import PocketLigandPairDataset

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    index_path = config.index
    pocket_type = config.type

    assert 'split' in config
    split_indices_dict = torch.load(config.split)
    train_split_indices = split_indices_dict['train']
    valid_split_indices = split_indices_dict['valid']
    test_split_indices = split_indices_dict['test']  # the 'key' of val_dataset is 'test'

    assert name == 'pl'
    dataset = PocketLigandPairDataset(root, index_path, pocket_type, *args, **kwargs)

    train_dataset = GetSubset(dataset, indices=train_split_indices)
    valid_dataset = GetSubset(dataset, indices=valid_split_indices)
    test_dataset = GetSubset(dataset, indices=test_split_indices)

    subsets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    return subsets

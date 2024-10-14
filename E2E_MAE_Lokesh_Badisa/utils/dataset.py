import os
import h5py
import math
import torch
import random
import numpy as np
from pathlib import Path
from einops import rearrange    
from bisect import bisect_left
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


BOOSTED_TOP_NEGLECT = [259173,
 259174,
 259175,
 259176,
 259177,
 259178,
 259179,
 259180,
 259181,
 259182,
 259183,
 259184,
 259185,
 259186,
 259187,
 259188,
 259189,
 259190,
 259191,
 259192,
 259193,
 259194,
 259195,
 259196,
 259197,
 259198,
 259199,
 451169,
 451170,
 451171,
 451172,
 451173,
 451174,
 451175,
 451176,
 451177,
 451178,
 451179,
 451180,
 451181,
 451182,
 451183,
 451184,
 451185,
 451186,
 451187,
 451188,
 451189,
 451190,
 451191,
 451192,
 451193,
 451194,
 451195,
 451196,
 451197,
 451198,
 451199]

QUARK_GLUON_NEGLECT = []

class H5Dataset(Dataset):
    '''
    Loads a dataset from h5 file
    args:
        file_path: str, path to the h5 file
        partition: str, one of 'train', 'validation', or 'test'
    '''
    def __init__(self, 
                 file_path: str, 
                 partition: str
    ) -> None:
        assert partition in ['train', 'validation', 'test'],\
              "Partition must be one of 'train', 'validation', or 'test'"
        self.file_path = file_path
        self.data = h5py.File(file_path, 'r')[f'{partition}_jet']          
        if Path(file_path).parent.stem == 'QG':
            self.labels = h5py.File(file_path, 'r')[f'{partition}_meta'][:,-1]
            self.neglect = np.array(QUARK_GLUON_NEGLECT)
        else:
            self.labels = h5py.File(file_path, 'r')[f'{partition}_meta'][:,0]
            self.neglect = np.array(BOOSTED_TOP_NEGLECT)

    def __len__(self):
        return self.labels.shape[0] - len(self.neglect)

    def __getitem__(self, idx):
        idx = idx + bisect_left(self.neglect, idx)
        self.data[idx][self.data[idx] < 1e-3] = 0
        return self.data[idx], self.labels[idx]

class ChunkedDistributedSampler(Sampler):
    def __init__(self, total_samples, chunk_size=32, shuffle=True, num_replicas=None, rank=None):
        indices = list(range(total_samples))
        random.shuffle(indices)
        self.data_source = indices
        self.chunk_size = chunk_size
        self.num_chunks = total_samples // chunk_size
        self.indices = list(range(total_samples))
        self.shuffle = shuffle
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()

        # Ensure that all replicas have the same number of samples
        indices = self.indices + self.indices[:(self.total_size - len(self.indices))]
        assert len(indices) == self.total_size

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples


class RegressionDataset(Dataset):
    def __init__(self, h5_path, partition='train', transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file[f'{partition}_jet']
        self.labels = self.h5_file[f'{partition}_meta']
        self.dataset_size = self.labels.shape[0]

        self.chunk_size = 32

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data), torch.from_numpy(labels)

    def __del__(self):
        self.h5_file.close()



def prepare_dataloader(data_dir: str, batch_size: int):
    trainset = H5Dataset(data_dir, 'train')
    valset = H5Dataset(data_dir, 'validation')
    testset = H5Dataset(data_dir, 'test')
    print("Data Loaded")
    WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
    GLOBAL_RANK = int(os.environ['SLURM_PROCID'])


    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(trainset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK)
        # sampler = DistributedSampler(trainset)
    )
    val_loader = DataLoader(
        valset,
        batch_size=512,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(valset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=False)
        # sampler = DistributedSampler(valset)
    )
    test_loader = DataLoader(
        testset,
        batch_size=512,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(testset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=False)
    )
    return train_loader, val_loader, test_loader

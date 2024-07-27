import os
import h5py
import math
import torch
import random
import numpy as np
from einops import rearrange    
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


def collate(batch):
    data, labels = list(zip(*batch))
    data = torch.stack(data)
    data = rearrange(data, 'b h w c-> b c h w')
    return data, labels


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
        self.labels = h5py.File(file_path, 'r')[f'{partition}_meta']
        self.neglect = []
        for i, d in enumerate(self.data):
            if np.min(d) == np.max(d):
                self.neglect.append(i)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))    
        return torch.Tensor(data), torch.Tensor(np.array(self.labels[idx][-1]))

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


def prepare_dataloader(data_dir: str, batch_size: int):
    trainset = H5Dataset(data_dir, 'train')
    valset = H5Dataset(data_dir, 'validation')
    WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        sampler=ChunkedDistributedSampler(len(trainset), num_replicas=WORLD_SIZE, rank=LOCAL_RANK)
    )
    val_loader = DataLoader(
        valset,
        batch_size=512,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        sampler=ChunkedDistributedSampler(len(trainset), num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=False)
    )
    return train_loader, val_loader

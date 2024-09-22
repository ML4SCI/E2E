import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Subset

_GLOBAL_SEED = 0
logger = getLogger()


def make_gsoc_dataset(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    dataset = GsocDataset3( root_path, preload_size=batch_size)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = ChunkedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    val_sampler = ChunkedSampler(val_indices, chunk_size=batch_size, shuffle=False)

    train_data_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   pin_memory=pin_mem,
                                   collate_fn=collator,
                                   num_workers=num_workers)

    val_data_loader = DataLoader(dataset,
                                 batch_size=batch_size, 
                                 sampler=val_sampler, 
                                 pin_memory=pin_mem,
                                 collate_fn=collator, 
                                 num_workers=num_workers)
    
    logger.info('GSOC unsupervised data loaders created')

    return dataset, train_data_loader, val_data_loader


class ChunkedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = list(range(len(data_source)))
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        np.random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class GsocDataset3(torch.utils.data.Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['jet']
        #self.labels = self.h5_file['m0']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

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
            #self.preloaded_labels = self.labels[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        #labels = self.preloaded_labels[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data)#, torch.from_numpy(labels)

    def __del__(self):
        self.h5_file.close()

        
def make_gsoc_dataset_iris(
    batch_size,
    split_size=None,
    chunk_size=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    # Instantiate the dataset
    # mode can be 'train', 'test', or 'validation' depending on what you're doing
    train_dataset = Dataset4(file_path=root_path, mode='train', chunk_size=chunk_size)
    val_dataset = Dataset4(file_path=root_path, mode='validation', chunk_size=chunk_size)
    
    train_length = len(train_dataset)
    val_length = len(val_dataset)
    train_indices = list(range(int(train_length*split_size/100)))
    val_indices = list(range(int(val_length*50/100)))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Create the DataLoaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collator,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    # Create the DataLoader
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collator,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    logger.info('GSOC unsupervised data loaders created')

    return train_dataset, train_data_loader, val_data_loader

class Dataset4(torch.utils.data.Dataset):
    """Dataset Class"""

    def __init__(self, file_path,mode,chunk_size = 32):
        """
        Arguments:
            file_path (string): Path to the HDF5 file
            mode (string): "train", "test" or "validation" set to choose from.
            chunk_size: The chunk size to read the data from.
        """
        self.file_path = file_path
        self.mode = mode
        self.chunk_size = chunk_size

        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[f"{self.mode}_jet"]) // self.chunk_size

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.file = h5py.File(self.file_path, 'r')

    def __getitem__(self, idx: int):

        if not hasattr(self, 'file'):
            self.open_hdf5()

        # Here idx is the chunk ID

        imgs = torch.tensor(self.file[f'{self.mode}_jet'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...].transpose(0,3,1,2))
        labels = torch.tensor(self.file[f'{self.mode}_meta'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...])
        return imgs, labels

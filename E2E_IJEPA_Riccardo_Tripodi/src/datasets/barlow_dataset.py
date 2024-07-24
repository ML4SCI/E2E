import numpy as np

from logging import getLogger

import torch
import h5py
from torch.utils.data import DataLoader

_GLOBAL_SEED = 0
logger = getLogger()


def make_barlow_dataset(
    transforms_list,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
):
    
    dataset = GsocDataset3(root_path, preload_size=batch_size, transforms_list=transforms_list)

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
                                   #collate_fn=collator,
                                   num_workers=num_workers)

    val_data_loader = DataLoader(dataset,
                                 batch_size=batch_size, 
                                 sampler=val_sampler, 
                                 pin_memory=pin_mem,
                                 #collate_fn=collator, 
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
    def __init__(self, h5_path, transforms_list=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms_list = transforms_list
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
        data1 = torch.from_numpy(self.preloaded_data[local_idx])
        data2 = data1.clone()
        #labels = self.preloaded_labels[local_idx]
        if self.transforms_list:
            data1 = self.transforms_list(data1)
            data2 = self.transforms_list(data2)
        
        return data1, data2

    def __del__(self):
        self.h5_file.close()

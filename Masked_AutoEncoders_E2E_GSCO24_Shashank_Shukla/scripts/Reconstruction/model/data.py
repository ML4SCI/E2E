from util import *
#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------Dataset------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

class ChunkedDistributedSampler(Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False, num_replicas=None, rank=None):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = list(range(len(data_source)))
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

        # Subsample for the current process (rank)
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples
    
class H5MaskedAutoEncoderDataset(Dataset):
    def __init__(self, h5_path, preload_size=3200):
        self.h5_path = h5_path
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['train_jet']
        self.dataset_size = self.data.shape[0]
        self.chunk_size = self.data.chunks
        self.preloaded_data = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        sample = torch.from_numpy(data).permute(2,0,1)/255.0
        if sample.max() > 1.01:
            with open(f'logs.txt', 'a') as f:
                f.write(f'Found the issue....\n')
                
        return {"img":sample} 

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()
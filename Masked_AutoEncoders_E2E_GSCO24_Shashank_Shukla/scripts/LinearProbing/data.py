from util import *
#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------Dataset------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

class ChunkedDistributedSampler(Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False, num_replicas=None, rank=None):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = sorted(data_source)
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
    def __init__(self, h5_path, mode = 'train', n_samples = -1, preload_size=3200):
        self.h5_path = h5_path
        self.n_samples = n_samples
        self.preload_size = preload_size
        
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file[f'{mode}_jet']
        self.label = self.h5_file[f'{mode}_meta']
        
        self.dataset_size = self.data.shape[0]
        self.chunk_size = self.data.chunks
        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size if self.n_samples == -1 else self.n_samples

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.dataset_size}")

        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.label[preload_start:preload_end][:,0]
        
        local_idx = idx - self.preload_start
        if local_idx >= len(self.preloaded_data):
            raise IndexError(f"Local index {local_idx} out of bounds for preloaded data with size {len(self.preloaded_data)}")
        data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        
        sample = torch.from_numpy(data).permute(2,0,1)/255.0
        labels = torch.tensor(labels)
                
        return {"img":sample, "labels": labels} 

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()
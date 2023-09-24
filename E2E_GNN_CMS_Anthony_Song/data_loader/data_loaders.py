import torch_geometric
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TopGunDataLoader(BaseDataLoader):
    '''
    Data loader for top gun dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size, multi_gpu, training=True):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.test_dset = test_dset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.multi_gpu = multi_gpu
        super().__init__(self.train_dset, batch_size, shuffle, validation_split, num_workers)
        

    def get_data_loader(self):
        if self.multi_gpu:
            loader_type = torch_geometric.data.DataListLoader
        else:
            loader_type = torch_geometric.data.DataLoader
        
        train_loader = loader_type(
            self.train_dset, shuffle=True, batch_size=self.train_batch_size, pin_memory=True, num_workers=0
        )
        val_loader = loader_type(
            self.val_dset, shuffle=False, batch_size=self.val_batch_size, pin_memory=True, num_workers=0
        )
        test_loader = loader_type(
            self.test_dset, shuffle=False, batch_size=self.test_batch_size, num_workers=0
        )
        return train_loader, val_loader, test_loader
    

class QGDataLoader():
    '''
    Data loader for quark gluon dataset
    '''
    def __init__(self, train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size, training=True):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.test_dset = test_dset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        

    def get_data_loader(self):
        train_loader = DataLoader(self.train_dset, batch_size=self.train_batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dset, batch_size=self.val_batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dset, batch_size=self.test_batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
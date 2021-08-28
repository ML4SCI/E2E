"""Define the Dataloaders"""

from torch.utils.data import Dataset,ConcatDataset,DataLoader,SubsetRandomSampler
import pyarrow.parquet as pq
import json
import numpy as np
from utils import transform_y

params = json.load(open("./E2E/E2E_Regression_Anis_Ismail/experiment.json",'r'))

class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        self.cols = ["X_jet", "genM", "iphi", "ieta"]
        self.label = label

    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pandas()
        data_dict = {'X_jet': np.array([[[coord for coord in xk] for xk in xj] for xj in data["X_jet"].values[0]],
                                       ndmin=3, dtype=np.float32)}
        data_dict['X_jet'][0] = params["channel1_scale"] * data_dict['X_jet'][0]
        data_dict['X_jet'][1] = params["channel2_scale"] * data_dict['X_jet'][1]
        data_dict['X_jet'][2] = params["channel3_scale"] * data_dict['X_jet'][2]
        data_dict['genM'] = transform_y(np.float32(data['genM'].values))
        data_dict['iphi'] = np.float32(data['iphi'].values) / 360.
        data_dict['ieta'] = np.float32(data['ieta'].values) / 170.
        # Zero-Suppression
        data_dict['X_jet'][data_dict['X_jet'] < 1.e-3] = 0.
        # High Value Suppression
        data_dict['X_jet'][0][data_dict['X_jet'][0] > 50] = 1.
        data_dict['X_jet'][1][data_dict['X_jet'][1] > 5] = 1.
        data_dict['X_jet'][2][data_dict['X_jet'][2] > 5] = 1.
        return data_dict

    def __len__(self):
        return self.parquet.num_row_groups


def train_val_loader(datasets, batch_size, random_sampler=True):
    dset = ConcatDataset([ParquetDataset(dataset, datasets.index(dataset)) for dataset in datasets])
    idxs = np.random.permutation(len(dset))
    print(len(dset))
    train_cut = int(len(dset) * 0.9)
    val_cut = int(len(dset) * 0.05)
    if random_sampler:
        train_sampler = SubsetRandomSampler(idxs[:train_cut])
        val_sampler = SubsetRandomSampler(idxs[train_cut:train_cut + val_cut])
        test_sampler = SubsetRandomSampler(idxs[train_cut + val_cut:])
    else:
        train_sampler, val_sampler, test_sampler = None, None, None
    train_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=train_sampler,
                              pin_memory=True)
    val_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=val_sampler,
                            pin_memory=True)
    test_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=test_sampler,
                             pin_memory=True)
    return train_loader, val_loader, test_loader

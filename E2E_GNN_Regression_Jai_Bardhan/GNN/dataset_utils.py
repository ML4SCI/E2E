import torch
import numpy as np
import torch_geometric

def get_loaders(train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size, multi_gpu):
    '''
        This function provides the loaders for the datasets
        
        Args:
            train_dset: Training dataset
            val_dset: Validation dataset
            test_dset: Test dataset
            train_batch_size: Training batch size
            val_batch_size: Validation batch size
            test_batch_size: Test batch size

        Returns:
            train_loader: Training dataset data loader
            val_loader: Validation dataset data loader
            test_loader: Test dataset data loader
    '''
    if multi_gpu:
        loader_type = torch_geometric.data.DataListLoader
    else:
        loader_type = torch_geometric.data.DataLoader
    train_loader = loader_type(
        train_dset, shuffle=True, batch_size=train_batch_size, pin_memory=True, num_workers=0
    )
    val_loader = loader_type(
        val_dset, shuffle=False, batch_size=val_batch_size, pin_memory=True, num_workers=0
    )
    test_loader = loader_type(
        test_dset, shuffle=False, batch_size=test_batch_size, num_workers=0
    )
    return train_loader, val_loader, test_loader


def positional_encoding(data, pe_scales):
    '''
        Performs cos/sin positional encoding provided the data and the scales
        Args:
            data: The data on which the positional encoding has to be applied
            pe_scales: The scales of the positional encoding

        Returns:
            The tensor with the cos/sin positional encoding performed on the data 
    '''
    pe_cos = torch.cat([torch.cos(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)
    pe_sin = torch.cat([torch.sin(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)

    output= torch.cat([torch.as_tensor(data), pe_cos, pe_sin], dim=1)
    return output


def normalize_x(x):
    '''
        Performs normalization of input data
        Args:
            x: The data be normalized

        Returns:
            The normalized tensor x
    '''
    x = x - np.array([0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754])
    x = x / np.array([10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])

    return x

def points_all_channels(X_jets, suppression_thresh):
    idx = np.where(abs(X_jets).sum(axis=0) > suppression_thresh)
    pos = np.array(idx).T / X_jets.shape[1]
    x = X_jets[:, idx[0], idx[1]].T

    return x, pos

def points_channel_wise(X_jets, suppression_thresh):
    idx = np.where(abs(X_jets) > suppression_thresh)
    total_pos = np.array(idx).T
    
    channel_pos = total_pos[:, 0]
    channel_onehot = np.eye(X_jets.shape[0])[channel_pos]

    xy_pos = total_pos[:, 1:] / X_jets.shape[1:]

    pos = np.concatenate([xy_pos, channel_onehot], axis=1)
    x = X_jets[idx[0], idx[1], idx[2]].T 
    x = np.expand_dims(x, axis=1)
    
    return x, pos
    

def edge_features_as_R(pos, edge_index):
    pos_i = pos[edge_index[0]]
    pos_j = pos[edge_index[1]]

    R = torch.sqrt(torch.relu(
        ((pos_i - pos_j) ** 2).sum(dim=-1, keepdims=True) 
    ))

    return R

def edge_feat_as_diff(x, edge_index):
    x_i = x[edge_index[0]]
    x_j = x[edge_index[1]]

    edge_attr = (x_i - x_j)

    return edge_attr
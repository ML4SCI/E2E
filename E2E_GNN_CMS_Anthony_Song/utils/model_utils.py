import os
import torch
import torch_geometric
from tqdm.auto import tqdm

def edge_features_as_R(pos, edge_index):
    pos_i = pos[edge_index[0]]
    pos_j = pos[edge_index[1]]

    R = torch.sqrt(torch.relu(
        ((pos_i - pos_j) ** 2).sum(dim=-1, keepdims=True) 
    ))

    return R


def compute_degree(train_dset, k=7, device='cpu', force_recompute=False):

    if os.path.exists('deg.pt') and not force_recompute:
        deg = torch.load('deg.pt')
    else:
        max_degree = k + 1
        deg = torch.zeros(max_degree + 1, dtype=torch.long, device=device)
        for data in tqdm(train_dset, desc='Degree Distribution'):
            data = data.to(device, non_blocking=True)
            edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, num_workers=1)
            d = torch_geometric.utils.degree(edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        torch.save(deg, 'deg.pt')
    return deg # tensor([       0,        0,        0,        0,        0,        0,        0, 64694479, 23216198])
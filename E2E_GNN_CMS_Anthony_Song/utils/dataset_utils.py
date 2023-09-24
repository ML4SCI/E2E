import torch
import numpy as np
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj, to_undirected


def get_loaders(train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size, multi_gpu):
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
    pe_cos = torch.cat([torch.cos(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)
    pe_sin = torch.cat([torch.sin(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)

    output= torch.cat([torch.as_tensor(data), pe_cos, pe_sin], dim=1)
    return output


def normalize_x(x):
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


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


# LapPE
def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):

    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs



# RWSE thing
def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def compute_enc_transform(x, edge_index, transform_flags):

    N = x.shape[0]

    to_return = {}
    
    if transform_flags["LapPE"]:
        undir_edge_idx = to_undirected(edge_index, num_nodes  = N)
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_idx, normalization=transform_flags["LapPEnorm"], num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray()) 
        max_freqs = transform_flags["LapPEmax_freq"]
        eigvec_norm = transform_flags["LapPEeig_norm"]

        EigVals, EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm
        )

        to_return['eigvals'] = EigVals
        to_return['eigvecs'] = EigVecs

    if transform_flags["RWSE"]:
        kernel_param_times = transform_flags["RWSEkernel_times"]
        rw_landing = get_rw_landing_probs(ksteps=kernel_param_times,
                                          edge_index=edge_index,
                                          num_nodes=N)
        to_return['rwse'] = rw_landing

    
    return to_return
import torch_geometric
import torch
from utils.model_utils import edge_features_as_R
from base import BaseModel


class SimpleGAT(BaseModel):
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.k = k
        self.edge_feat = edge_feat
        self.args = args
        if self.edge_feat == 'none':
            edge_dim = None
        elif self.edge_feat == 'R':
            edge_dim = 1
        else:
            raise NotImplementedError(f"Edge feat {self.edge_feat} not implemented")
        self.gat_conv_1 = torch_geometric.nn.GATv2Conv(
            in_channels=x_size if not use_pe else x_size * (pe_scales * 2 + 1),
            out_channels=16,
            heads=4,
            edge_dim=edge_dim
        )
        self.bn_1 = torch_geometric.nn.BatchNorm(64)
        self.gat_conv_2 = torch_geometric.nn.GATv2Conv(
            in_channels=16 * 4,
            out_channels=32,
            heads=4,
            edge_dim=edge_dim
        )
        self.bn_2 = torch_geometric.nn.BatchNorm(128)
        self.act = torch.nn.ReLU()

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        x = data.x

        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch)
        if self.edge_feat == 'none':
            edge_attr = None
        elif self.edge_feat == 'R':
            edge_attr = edge_features_as_R(pos, edge_index)
        else:
            raise NotImplementedError(f"Edge feat {self.edge_feat} is not implemented")

        x_out = self.act(self.bn_1(self.gat_conv_1(
            x, edge_index, edge_attr=edge_attr)))
        x_out = self.act(self.bn_2(self.gat_conv_2(
            x_out, edge_index, edge_attr=edge_attr)))

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return x_out
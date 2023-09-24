import torch_geometric
import torch
from base import BaseModel

class GatedGCNNet(BaseModel):
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.k = k
        self.use_pe = use_pe
        self.args = args
        input_size = x_size if not self.use_pe else x_size * (pe_scales * 2 + 1)
        self.gated_conv_1 = torch_geometric.nn.ResGatedGraphConv(
            in_channels=input_size,
            out_channels=64,
        )
        self.bn_1 = torch_geometric.nn.BatchNorm(64)
        self.gated_conv_2 = torch_geometric.nn.ResGatedGraphConv(
            in_channels=64,
            out_channels=128,
        )
        self.bn_2 = torch_geometric.nn.BatchNorm(128)
        self.act = torch.nn.ReLU()

    def forward(self, data):
        pos = data.pos
        x = data.x
        batch = data.batch

        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch)

        x_out = self.act(self.bn_1(self.gated_conv_1(x, edge_index)))
        x_out = self.act(self.bn_2(self.gated_conv_2(x_out, edge_index)))

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return x_out

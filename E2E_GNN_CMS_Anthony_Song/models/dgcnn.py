import torch_geometric
import torch
from models.layers import MLPStack
from base import BaseModel

class DynamicEdgeConvPN(torch.nn.Module):
    def __init__(self, edge_nn, nn, k=7, edge_feat='none', aggr='max', flow='source_to_target') -> None:
        super().__init__()
        self.nn = nn
        self.k = k
        self.edge_conv = torch_geometric.nn.EdgeConv(nn=edge_nn, aggr=aggr)
        self.flow = flow
        self.edge_feat = edge_feat

    def forward(self, x, pos, batch):
        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch, flow=self.flow)

        edge_out = self.edge_conv(x, edge_index)

        x_out = self.nn(x)

        return edge_out + x_out


class DGCNN(BaseModel):
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.args = args
        self.dynamic_conv_1 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [x_size * 2 if not use_pe else x_size * 2 * (pe_scales * 2 + 1), 32, 32, 32], bn=True, act=True
            ),
            nn=MLPStack(
                [x_size if not use_pe else x_size * (pe_scales * 2 + 1), 32, 32, 32], bn=True, act=True
            ),
            k=k,
            edge_feat=edge_feat
        )

        self.dynamic_conv_2 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [64, 64, 64, 128], bn=True, act=True
            ),
            nn=MLPStack(
                [32, 64, 64, 128], bn=True, act=True
            ),
            k=k,
            edge_feat=edge_feat
        )

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        x_out = self.dynamic_conv_1(
            x, pos, batch
        )
        x_out = self.dynamic_conv_2(
            x_out, x_out, batch
        )

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return x_out


import torch_geometric
import torch

class MLPStack(torch.nn.Module):
    '''
        A simple MLP stack that stacks multiple linear-bn-act layers
    '''
    def __init__(self, layers, bn=True, act=True, p=0):
        super().__init__()
        assert len(layers) > 1, "At least input and output channels must be provided"

        modules = []
        for i in range(1, len(layers)):
            modules.append(
                torch.nn.Linear(layers[i-1], layers[i])
            )
            modules.append(
                torch.nn.BatchNorm1d(layers[i]) if bn == True else torch.nn.Identity()
            )
            modules.append(
                torch.nn.SiLU() if bn == True else torch.nn.Identity()
            )
            modules.append(
                torch.nn.Dropout(p=p) if p != 0 else torch.nn.Identity()
            )

        self.mlp_stack = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp_stack(x)


class ResEdgeConv(torch.nn.Module):
    '''
        Internal convolution DynamicEdgeConv block inspired from ParticleNet
    '''
    def __init__(self, edge_nn, nn, k=7, edge_feat='none', aggr='max', flow='source_to_target') -> None:
        super().__init__()
        self.nn = nn
        self.k = k
        self.edge_conv = torch_geometric.nn.EdgeConv(nn=edge_nn, aggr=aggr)
        self.flow = flow
        self.edge_feat = edge_feat

    def forward(self, x, edge_index):
        edge_out = self.edge_conv(x, edge_index)
        x_out = self.nn(x)

        return edge_out + x_out


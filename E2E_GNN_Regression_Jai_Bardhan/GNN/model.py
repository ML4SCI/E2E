import torch_geometric
import torch
from tqdm.auto import tqdm
from gps_layer import GPSLayer
from layers import MLPStack, ResEdgeConv
import os
from dataset_utils import edge_features_as_R

class DynamicEdgeConvPN(torch.nn.Module):
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

    def forward(self, x, pos, batch):
        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch, flow=self.flow)

        edge_out = self.edge_conv(x, edge_index)

        x_out = self.nn(x)

        return edge_out + x_out


class DGCNN(torch.nn.Module):
    '''
        DGCNN network that is similar to the ParticleNet architecture
    '''
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        '''
            Init fn. of DGCNN backbone
            Args:
                args: ArgumentParser parsed args
                x_size: Input dimension of data.x
                pos_size: Input dimension of data.pos
                edge_feat: Which edge feature to use 
                k: K from K-NN of the data to form the graph
                use_pe: Whether to use positional encoding
                pe_scales: The number of scales for positional encoding
        '''

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

class GatedGCNNet(torch.nn.Module):
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        '''
            Init fn. of GatedGCN backbone
            Args:
                args: ArgumentParser parsed args
                x_size: Input dimension of data.x
                pos_size: Input dimension of data.pos
                edge_feat: Which edge feature to use 
                k: K from K-NN of the data to form the graph
                use_pe: Whether to use positional encoding
                pe_scales: The number of scales for positional encoding
        '''
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


def compute_degree(train_dset, k=7, device='cpu', force_recompute=False):
    '''
        Computes the degree of the data given a (batched) iterator over the dataset.
        Args:
            train_dset: Iterator over the training dataset
            k: k from K-NN used for forming the graph from the data
            device: Device to return the tensor on
            force_recompute: Whether to recompute the degrees if a saved copy is available 
    '''    
    # Code from the PyG repository for computing for random graphs. Here graphs are constructed as k-nn
    # max_degree = -1
    # for data in tqdm(train_dset, desc='Max Degree'):
    #     edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, num_workers=1)
    #     d = torch_geometric.utils.degree(edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     max_degree = max(max_degree, d.max())

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

class PNANet(torch.nn.Module):
    def __init__(self, args, x_size, pos_size, deg, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        '''
            Init fn. of PNANet backbone
            Args:
                args: ArgumentParser parsed args
                x_size: Input dimension of data.x
                pos_size: Input dimension of data.pos
                deg: The degree of the nodes of the graph
                edge_feat: Which edge feature to use 
                k: K from K-NN of the data to form the graph
                use_pe: Whether to use positional encoding
                pe_scales: The number of scales for positional encoding
        '''

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
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.pna_conv_1 = torch_geometric.nn.PNAConv(
            in_channels=x_size if not use_pe else x_size * (pe_scales * 2 + 1),
            out_channels=64,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn1 = torch_geometric.nn.BatchNorm(64)

        self.pna_conv_2 = torch_geometric.nn.PNAConv(
            in_channels=64,
            out_channels=128,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn2 = torch_geometric.nn.BatchNorm(128)

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

        x_out = self.act(self.bn1(self.pna_conv_1(x, edge_index, edge_attr=edge_attr)))
        x_out = self.act(self.bn2(self.pna_conv_2(x_out, edge_index, edge_attr=edge_attr)))

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch) # Consider global add pool
        
        return x_out


class SimpleGAT(torch.nn.Module):
    '''
        Simple 2 layered GAT GNN
    '''
    def __init__(self, args, x_size, pos_size, edge_feat='none', k=7, use_pe=False, pe_scales=0):
        '''
            Init fn. of SimpleGAT backbone
            Args:
                args: ArgumentParser parsed args
                x_size: Input dimension of data.x
                pos_size: Input dimension of data.pos
                edge_feat: Which edge feature to use 
                k: K from K-NN of the data to form the graph
                use_pe: Whether to use positional encoding
                pe_scales: The number of scales for positional encoding
        '''
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


class RegressModel(torch.nn.Module):
    '''
        Model to perform the regression on the data. 
        Builds a small MLP network on a provided backbone network.
    '''
    def __init__(self, model, in_features, use_pe=False, pe_scales=0, predict_bins=False, num_bins=10):
        '''
            Init fn. of the RegressModel.
            Args:
                model: The backbone model to operate on the images
                in_features: The size of the output of the backbone
                use_pe: Whether positional encoding is being used
                pe_scales: The scales of the positional encoding
        '''
        super().__init__()
        self.model = model
        self.predict_bins = predict_bins

        self.out_mlp = MLPStack(
            [in_features + 3, in_features * 2, in_features * 2, in_features, in_features // 2],
            bn=True, act=True
        )

        self.out_regress = torch.nn.Linear(in_features//2, 1)

        if self.predict_bins:
            self.out_pred = torch.nn.Linear(in_features // 2, num_bins)

    def forward(self, data):
        return_dict = {}

        out = self.model(data)
        out = torch.cat(
            [out, data.pt.unsqueeze(-1), data.ieta.unsqueeze(-1), data.iphi.unsqueeze(-1)], dim=1
        )
        out = self.out_mlp(out)
        regress_out = self.out_regress(out)

        return_dict['regress'] = regress_out
        
        if self.predict_bins:
            pred_out = self.out_pred(out)
            return_dict['class'] = pred_out

        return return_dict


class GPSModel(torch.nn.Module):
    '''
        GraphGPS model from the paper
    '''

    def __init__(self, args, x_size, pos_size, dim_h, k=7, deg=None, use_pe=False, pe_scales=10, predict_bins=False, num_bins=10):
        '''
            Init fn. of GPSModel
            Args:
                args: ArgumentParser parsed args
                x_size: Input dimension of data.x
                pos_size: Input dimension of data.pos
                dim_h: Output dimension 
                k: K from K-NN of the data to form the graph
                deg: Degree of nodes of the data (Used for PNAConv)
                use_pe: Whether to use positional encoding
                pe_scales: The number of scales for positional encoding
                predict_bins: Whether to add the additional task of predicting the bins
                num_bins: Number of bins to be predicted in the additional task
        '''
        super().__init__()

        self.num_gps_layers = args.num_gps_layers
        self.k = k
        self.dim_h = dim_h
        self.predict_bins = predict_bins

        input_size = x_size if not use_pe else x_size * (pe_scales * 2 + 1)

        gps_layers = []
        assert self.num_gps_layers > 0, "At least one GPS layer must be used"

        gps_layers.append(
            GPSLayer(
                args,
                input_size,
                dim_h,
                args.gps_mpnn_type,
                args.gps_global_type,
                args.gps_num_heads,
                deg=deg,    
            )
        )

        for _ in range(1, self.num_gps_layers):
            gps_layers.append(
                GPSLayer(
                    args,
                    dim_h,
                    dim_h,
                    args.gps_mpnn_type,
                    args.gps_global_type,
                    args.gps_num_heads,
                    deg=deg,
                )
            )

        self.gps_layers = torch.nn.Sequential(*gps_layers)

        self.out_mlp = MLPStack([dim_h + 3, dim_h * 2, dim_h])
        self.out_regress = torch.nn.Linear(dim_h, 1)

        if self.predict_bins:
            self.out_pred = torch.nn.Linear(dim_h, num_bins)

    def forward(self, batch):
        return_dict = {}

        batch.edge_index = torch_geometric.nn.knn_graph(x=batch.pos, k=self.k, batch=batch.batch)

        batch_out = self.gps_layers(batch)

        x_out = torch_geometric.nn.global_mean_pool(batch_out.x, batch_out.batch)
        
        x_out = torch.cat(
            [x_out, batch.pt.unsqueeze(-1), batch.ieta.unsqueeze(-1), batch.iphi.unsqueeze(-1)], dim=-1
        )
        x_out = self.out_mlp(x_out)

        regress_out = self.out_regress(x_out)
        return_dict['regress'] = regress_out

        if self.predict_bins:
            pred_out = self.out_pred(x_out)
            return_dict['class'] = pred_out

        return return_dict

def get_model(args, device, model, point_fn, edge_feat, train_loader, multi_gpu=False, pretrained=False, use_pe=False, pe_scales=0, predict_bins=False, num_bins=10):
    '''
        Returns the model based on the arguments
        Args:
            args: The argparse argument
            device: The device to run the model on
            model: The backbone model choice
            pretrained: Whether to use the pretrained backbone
            use_pe: Whether positional encoding is being used
            pe_scales: The scale of the positional encoding

        Returns:
            regress_model: Model that is used to perform regression
    '''
    if point_fn == 'total':
        x_size = 10
        pos_size = 2
    elif point_fn == 'channel_wise':
        x_size = 11
        pos_size = 10
    else:
        raise NotImplementedError()

    if args.LapPE:
        x_size += (args.LapPEmax_freq * 2)
    if args.RWSE:
        x_size += len(args.RWSEkernel_times)

    if model == 'dgcnn':
        input_model = DGCNN(args, x_size=x_size, pos_size=pos_size, edge_feat=edge_feat, use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'gat':
        input_model = SimpleGAT(args, x_size=x_size, pos_size=pos_size, edge_feat=edge_feat, use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'pna':
        deg = compute_degree(train_loader, device=device)
        input_model = PNANet(args, x_size, pos_size, deg, edge_feat=edge_feat, use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'gatedgcn':
        input_model = GatedGCNNet(args, x_size, pos_size, edge_feat=edge_feat, use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'gps':
        if args.gps_mpnn_type == 'pnaconv':
            deg = compute_degree(train_loader, device=device)
        else:
            deg = None
        regress_model = GPSModel(
            args,
            x_size,
            pos_size,
            dim_h=args.gps_dim_h,
            k=7,
            deg=deg,
            use_pe=use_pe,
            pe_scales=pe_scales,
            predict_bins=predict_bins,
            num_bins=num_bins
        )

        if multi_gpu:
            regress_model = torch_geometric.nn.DataParallel(regress_model)

        regress_model = regress_model.to(device)

        return regress_model
    else:
        raise NotImplementedError(f"Model type {model} not implemented")

    regress_model = RegressModel(
        model=input_model,
        in_features=128,
        use_pe=use_pe,
        pe_scales=pe_scales,
        predict_bins=predict_bins,
        num_bins=num_bins,
    )

    if multi_gpu:
        regress_model = torch_geometric.nn.DataParallel(regress_model)

    regress_model = regress_model.to(device)

    return regress_model
import torch_geometric
import torch
from performer_pytorch import SelfAttention
from models.layers import MLPStack, ResEdgeConv
from base import BaseModel


class GPSLayer(torch.nn.Module):
    def __init__(self, args, input_size, dim_h, local_mpnn_type, global_model_type, num_heads, layer_norm=False, batch_norm=True, dropout=0., attn_dropout=0., deg=None, k=7):

        super().__init__()
        
        self.dim_h = dim_h
        self.local_mpnn_type = local_mpnn_type
        self.global_model_type = global_model_type
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.k = k

        if local_mpnn_type == None:
            self.local_model = None
        elif local_mpnn_type == 'gatedgcn':
            self.local_model = torch_geometric.nn.ResGatedGraphConv(in_channels=input_size, out_channels=dim_h)
        elif local_mpnn_type == 'gat':
            self.local_model = torch_geometric.nn.GATv2Conv(in_channels=input_size, out_channels=dim_h, heads=num_heads)
        elif local_mpnn_type == 'edgeconv':
            self.local_model = ResEdgeConv(nn=MLPStack([input_size, dim_h]), edge_nn=MLPStack([input_size * 2, dim_h]))            
        elif local_mpnn_type == 'pnaconv':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']

            self.local_model = torch_geometric.nn.PNAConv(
                in_channels=input_size,
                out_channels=dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False
            )
        else:
            raise NotImplementedError()

        if global_model_type == None:
            self.self_attn = None
        elif global_model_type == 'transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True
            )
        elif global_model_type == 'performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False 
            )
        else:
            raise NotImplementedError()


        if self.layer_norm and self.batch_norm:
            raise ValueError('Cannot apply two types of normalization together')

        # Normalization for MPNN and Self Attention Together
        if self.layer_norm:
            self.norm1_local = torch_geometric.norm.GraphNorm(dim_h)
            self.norm1_attn = torch_geometric.norm.GraphNorm(dim_h)

        if self.batch_norm:
            self.norm1_local = torch.nn.BatchNorm1d(dim_h)
            self.norm1_attn = torch.nn.BatchNorm1d(dim_h)
        self.dropout_local = torch.nn.Dropout(p=self.dropout)
        self.dropout_attn = torch.nn.Dropout(p=self.dropout)


        # Feed Forward block
        self.activation = torch.nn.ReLU()
        self.ff_linear1 = torch.nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = torch.nn.Linear(dim_h * 2, dim_h)
        
        if self.layer_norm:
            self.norm2 = torch_geometric.norm.GraphNorm(dim_h)

        if self.batch_norm:
            self.norm2 = torch.nn.BatchNorm1d(dim_h)

        self.ff_dropout1 = torch.nn.Dropout(p=self.dropout)
        self.ff_dropout2 = torch.nn.Dropout(p=self.dropout)

        self.res_lin = MLPStack([input_size, dim_h])


    def forward(self, batch):
        h = batch.x
        h_in1 = h

        h_in1 = self.res_lin(h_in1)

        h_out_list = []

        if self.local_model is not None:
            h_local = self.local_model(h, batch.edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)

            h_out_list.append(h_local)


        if self.self_attn is not None:
            h_dense, mask = torch_geometric.utils.to_dense_batch(h_in1, batch.batch)
            
            if self.global_model_type == 'transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn
            
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)


        h = sum(h_out_list)


        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch


    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]

        return x


    def _ff_block(self, x):
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x)) # NOTE: There is no activation here! 



class GPSModel(BaseModel):

    def __init__(self, args, x_size, pos_size, dim_h, k=7, deg=None, use_pe=False, pe_scales=10, predict_bins=False, num_bins=10):
        super().__init__()

        self.num_gps_layers = args["num_gps_layers"]
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
                args["gps_mpnn_type"],
                args["gps_global_type"],
                args["gps_num_heads"],
                deg=deg,    
            )
        )

        for _ in range(1, self.num_gps_layers):
            gps_layers.append(
                GPSLayer(
                    args,
                    dim_h,
                    dim_h,
                    args["gps_mpnn_type"],
                    args["gps_global_type"],
                    args["gps_num_heads"],
                    deg=deg,
                )
            )

        self.gps_layers = torch.nn.Sequential(*gps_layers)

        self.out_mlp = MLPStack([dim_h + 3, dim_h * 2, dim_h])
        self.out_regress = torch.nn.Linear(dim_h, 1)

        if self.predict_bins:
            # self.out_pred = torch.nn.Linear(dim_h, num_bins)
            self.out_pred1 = torch.nn.Linear(dim_h, 16)
            self.relu = torch.nn.ReLU()
            self.out_pred2 = torch.nn.Linear(16, 1)
            self.sigmoid = torch.nn.Sigmoid()


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
            pred_out = self.out_pred1(x_out)
            pred_out = self.relu(pred_out)
            pred_out = self.out_pred2(pred_out)
            # pred_out = self.sigmoid(pred_out)
            return_dict['class'] = pred_out.squeeze(1)

        return return_dict
import torch
import torch_geometric
from performer_pytorch import SelfAttention

from layers import ResEdgeConv, MLPStack

class GPSLayer(torch.nn.Module):
    '''
        Local MPNN + full graph attention x-former layer
    '''
    def __init__(self, args, input_size, dim_h, local_mpnn_type, global_model_type, num_heads, layer_norm=False, batch_norm=True, dropout=0., attn_dropout=0., deg=None, k=7):
        '''
            Init fn. of GPSLayer
            Args:
                args: ArgumentParser parsed args
                input_size: Input dimension of the data to be operated on
                dim_h: Output dimension 
                local_mpnn_type: Type of the GNN Layer to use for local Neural Network
                global_model_type: Type of Self Attention (Transformer) model for global Neural Netwokr
                num_heads: Number of heads in the MSA block
                layer_norm: Whether to perform layernorm 
                batch_norm: Whether to perform batchnorm
                dropout: Dropout probability on the output vector
                attn_dropout: Dropout probability on the self attention matrix
                deg: Degree of nodes of the data (Used for PNAConv)
                k: K from K-NN of the data to form the graph
        '''
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

        h_in1 = self.res_lin(h_in1) # For res connection make same size

        h_out_list = []
        # Local MPNN
        if self.local_model is not None:
            h_local = self.local_model(h, batch.edge_index) # NOTE: The edge index is already present
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local # Residual connection

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)

            h_out_list.append(h_local)


        # MHA Attention
        if self.self_attn is not None:
            h_dense, mask = torch_geometric.utils.to_dense_batch(h_in1, batch.batch) # Here the updated thing goes because we want to make them the same size
            
            if self.global_model_type == 'transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn # Residual connection
            
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global embedding (here by summing)
        h = sum(h_out_list)

        # Feed forward block
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch


    def _sa_block(self, x, attn_mask, key_padding_mask):
        '''
            Self Attn Block
        '''

        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]

        return x


    def _ff_block(self, x):
        '''
            Feedforward Block
        '''
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x)) # NOTE: There is no activation here! 



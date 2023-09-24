import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import auroc


class GCN(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.3):
        super().__init__()
        torch.manual_seed(123)
        self.conv1 = GraphConv(c_in, c_hidden)
        self.conv2 = GraphConv(c_hidden,2*c_hidden)
        self.conv3 = GraphConv(2*c_hidden, c_hidden)
        self.lin1 = Linear(c_hidden, 4*c_out)
        self.lin2 = Linear(4*c_out, c_out)
        self.dp_rate_linear = dp_rate_linear

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch) 

        # classifier
        x = F.dropout(x, p=self.dp_rate_linear, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dp_rate_linear, training=self.training)
        x = self.lin2(x)

        return x
    

class GraphLevelGNN(pl.LightningModule):
    
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GCN(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.auroc = auroc
        
    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch

        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        
        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        
        
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        data.y = data.y.to(torch.int64)
        auc = self.auroc(x, data.y, task = 'binary')
        return loss, acc, auc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=0) 
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, auc = self.forward(batch, mode="train")
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_auc', auc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, auc = self.forward(batch, mode="val")
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc, auc = self.forward(batch, mode="test")
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_auc', auc, prog_bar=True)


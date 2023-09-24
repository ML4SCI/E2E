import torch_geometric
import torch
from models.layers import MLPStack
from base import BaseModel

class RegressModel(BaseModel):
    def __init__(self, model, in_features, use_pe=False, pe_scales=0, predict_bins=False, num_bins=10):
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
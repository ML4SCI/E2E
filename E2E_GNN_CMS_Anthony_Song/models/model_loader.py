import torch_geometric

from models.dgcnn import DGCNN
from models.gat import SimpleGAT
from models.gated_gcn import GatedGCNNet
from models.gps import GPSModel
from models.pna import PNANet
from models.regresson import RegressModel
from utils.model_utils import compute_degree


class GNNModelLoader():
    def __init__(self, device, model, edge_feat, train_loader, point_fn, mode, multi_gpu=False, pretrained=False, use_pe=False, num_pe_scales=0, predict_bins=False, num_bins=10,**kwargs) -> None:
        self.device = device
        self.model = model
        self.edge_feat = edge_feat
        self.train_loader = train_loader
        self.point_fn = point_fn
        self.mode = mode
        self.pretrained = pretrained
        self.use_pe = use_pe
        self.pe_scales = num_pe_scales
        self.predict_bins = predict_bins
        self.num_bins = num_bins
        self.multi_gpu = multi_gpu
        self.args = kwargs

    def get_model(self):
        if self.point_fn == 'total':
            if self.mode == 'regression':
                x_size = 10
            elif self.mode == 'classification':
                x_size = 5
            pos_size = 2
        elif self.point_fn == 'channel_wise':
            x_size = 11
            pos_size = 10
        else:
            raise NotImplementedError()

        if self.args["LapPE"]:
            x_size += (self.args["LapPEmax_freq"] * 2)
        if self.args["RWSE"]:
            x_size += len(self.args["RWSEkernel_times"])

        if self.model == 'dgcnn':
            input_model = DGCNN(self.args, x_size=x_size, pos_size=pos_size, edge_feat=self.edge_feat, use_pe=self.use_pe, pe_scales=self.pe_scales)
        elif self.model == 'gat':
            input_model = SimpleGAT(self.args, x_size=x_size, pos_size=pos_size, edge_feat=self.edge_feat, use_pe=self.use_pe, pe_scales=self.pe_scales)
        elif self.model == 'pna':
            deg = compute_degree(self.train_loader, device=self.device)
            input_model = PNANet(self.args, x_size, pos_size, deg, edge_feat=self.edge_feat, use_pe=self.use_pe, pe_scales=self.pe_scales)
        elif self.model == 'gatedgcn':
            input_model = GatedGCNNet(self.args, x_size, pos_size, edge_feat=self.edge_feat, use_pe=self.use_pe, pe_scales=self.pe_scales)
        elif self.model == 'gps':
            if self.args["gps_mpnn_type"] == 'pnaconv':
                deg = compute_degree(self.train_loader, device=self.device)
            else:
                deg = None
            regress_model = GPSModel(
                self.args,
                x_size,
                pos_size,
                dim_h=self.args["gps_dim_h"],
                k=7,
                deg=deg,
                use_pe=self.use_pe,
                pe_scales=self.pe_scales,
                predict_bins=self.predict_bins,
                num_bins=self.num_bins
            )

            if self.multi_gpu:
                regress_model = torch_geometric.nn.DataParallel(regress_model)

            regress_model = regress_model.to(self.device)

            return regress_model
        else:
            raise NotImplementedError(f"Model type {self.model} not implemented")

        regress_model = RegressModel(
            model=input_model,
            in_features=128,
            use_pe=self.use_pe,
            pe_scales=self.pe_scales,
            predict_bins=self.predict_bins,
            num_bins=self.num_bins,
        )

        if self.multi_gpu:
            regress_model = torch_geometric.nn.DataParallel(regress_model)

        regress_model = regress_model.to(self.device)

        return regress_model
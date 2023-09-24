import os
import torch 
import torch_geometric
import numpy as np
import pyarrow.parquet as pq
from utils.dataset_utils import normalize_x, points_all_channels, points_channel_wise, positional_encoding, compute_enc_transform


hcal_scale  = 1
ecal_scale  = 0.1
pt_scale    = 0.01
dz_scale    = 0.05
d0_scale    = 0.1
m0_scale    = 85
m1_scale    = 415
p0_scale = 400
p1_scale = 600


class PointCloudFromParquetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        save_data,
        id,
        filename,
        transform_flags,
        mode, 
        point_fn='total',
        use_x_normalization=False,
        suppresion_thresh=0,
        scale_histogram=False,
        predict_bins=False,
        min_mass=0,
        max_mass=1,
        num_bins=10,
        use_pe=False,
        pe_scales=0,
        output_mean_scaling=False,
        output_mean_value=None,
        output_norm_scaling=False,
        output_norm_value=None,
        construction = 'knn',
        k = 20,
    ) -> None:

        super().__init__()

        self.id = id
        self.file = pq.ParquetFile(filename)
        self.root_dir = data_dir
        self.save_data = save_data
        self.transform_flags = transform_flags
        self.mode = mode
        self.suppression_thresh = suppresion_thresh
        self.point_fn = point_fn
        self.use_pe = use_pe
        self.scale_histogram = scale_histogram
        self.predict_bins = predict_bins
        self.num_bins = num_bins
        self.use_x_normalization = use_x_normalization
        self.pe_scales = pe_scales
        self.output_mean_scaling = output_mean_scaling
        self.output_mean_value = output_mean_value
        self.output_norm_scaling = output_norm_scaling
        self.output_norm_value = output_norm_value
        self.construction = 'knn',
        self.k = k
        
        if self.predict_bins:
            bin_size = (max_mass - min_mass) / num_bins
            self.bins = [min_mass + i * bin_size for i in range(num_bins)]
        else:
            self.bins = None

    def __getitem__(self, idx, ):

        if self.save_data and os.path.exists(os.path.join(self.save_data, f'{self.id}_{idx}.pt')):
            data = torch.load(os.path.join(self.save_data, f'{self.id}_{idx}.pt'))
        else:
            row = self.file.read_row_group(idx).to_pydict()
            
            if self.mode == 'classification':
                arr = np.array(row['X_jets'][0])
            elif self.mode == 'regression':
                arr = np.array(row['X_jet'][0])
            
            if self.point_fn == 'total':
                x, pos = points_all_channels(arr, self.suppression_thresh)
            elif self.point_fn == 'channel_wise':
                x, pos = points_channel_wise(arr, self.suppression_thresh)
            else:
                raise NotImplementedError(f"No function for {self.point_fn}")
            
            if self.use_x_normalization:
                x = normalize_x(x)

            pt = row['pt'][0]
            if self.mode == 'classification':
                ieta = 0
                iphi = 0
                m = row['m0'][0]
            elif self.mode == 'regression':
                ieta = row['ieta'][0]
                iphi = row['iphi'][0]
                m = row['m'][0]

            if self.scale_histogram:
                if self.mode == 'regression':
                    if self.point_fn == 'total':
                        x[:, 0] *= pt_scale
                        x[:, 1] *= dz_scale
                        x[:, 2] *= d0_scale
                        x[:, 3] *= ecal_scale
                        x[:, 4] *= hcal_scale
                        pt = (pt - p0_scale) / p1_scale
                        m = (m - m0_scale) if not self.output_mean_scaling else m
                        m = m / m1_scale if not self.output_norm_scaling else m
                        iphi = iphi / 360.
                        ieta = ieta / 140.

                        # High value suppression
                        x[:, 1][x[:, 1] < -1] = 0
                        x[:, 1][x[:, 1] > 1] = 0
                        x[:, 2][x[:, 2] < -1] = 0
                        x[:, 2][x[:, 2] > 1] = 0

                        # Zero suppression
                        x[:, 0][x[:, 0] < 1e-2] = 0.
                        x[:, 3][x[:, 3] < 1e-2] = 0.
                        x[:, 4][x[:, 4] < 1e-2] = 0.
                    else:
                        raise NotImplementedError(f"Histogram scaling for {self.point_fn} is not yet supported")
                elif self.mode == 'classification':
                    if self.point_fn == 'total':
                        x[:, 0] *= pt_scale
                        x[:, 1] *= dz_scale
                        x[:, 2] *= d0_scale
                        pt = (pt - p0_scale) / p1_scale
                        m = (m - m0_scale) if not self.output_mean_scaling else m
                        m = m / m1_scale if not self.output_norm_scaling else m
                        iphi = iphi / 360.
                        ieta = ieta / 140.

                        # High value suppression
                        x[:, 1][x[:, 1] < -1] = 0
                        x[:, 1][x[:, 1] > 1] = 0
                        x[:, 2][x[:, 2] < -1] = 0
                        x[:, 2][x[:, 2] > 1] = 0

                        # Zero suppression
                        x[:, 0][x[:, 0] < 1e-2] = 0.
                    else:
                        raise NotImplementedError(f"Histogram scaling for {self.point_fn} is not yet supported")

            x = np.concatenate([x, pos], axis=1)

            if self.output_mean_scaling:
                m = m - self.output_mean_value
            
            if self.output_norm_scaling:
                m  = m / self.output_norm_value

            m_class = -1
            

            pos = torch.as_tensor(pos, dtype=torch.float)
            x = torch.as_tensor(x)
            if self.transform_flags["LapPE"] or self.transform_flags["RWSE"]:
                if self.construction == 'knn':
                    edge_index = torch_geometric.nn.knn_graph(x=pos, k = self.k, num_workers=0)
                if self.construction == 'fc':
                    edge_index = torch_geometric.nn.radius_graph(x=pos, r = 2e31-1)
                if self.construction == 'epsilon':
                    edge_index = torch_geometric.nn.radius_graph(x=pos, r = 10)

                transforms = compute_enc_transform(x, edge_index, self.transform_flags)

            if self.transform_flags["LapPE"]:
                x = torch.cat([x, transforms['eigvecs'], transforms['eigvals'].squeeze(-1)], dim=-1)
            if self.transform_flags["RWSE"]:
                x = torch.cat([x, transforms['rwse']], dim=-1)

            x =  x if not self.use_pe else positional_encoding(x, self.pe_scales)

            y_class = -1
            if self.mode == 'regression':
                if self.predict_bins:
                    if self.predict_bins:
                        for it, bin_start in enumerate(self.bins):
                            if bin_start > m:
                                m_class = it - 1
                                break
                        if m_class == -1: #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                            m_class = self.num_bins - 1
                    y_class = m_class
                else:
                    y_class = None
            elif self.mode == 'classification':
                y_class = int(row['y'][0])

            data = torch_geometric.data.Data(
                pos=pos.float(),
                x=x.float(),
                pt=torch.as_tensor(pt).unsqueeze(-1),
                ieta=torch.as_tensor(ieta).unsqueeze(-1),
                iphi=torch.as_tensor(iphi).unsqueeze(-1),
                y=torch.as_tensor(m),
                y_class = y_class
            )

            if self.save_data:
                torch.save(data, os.path.join(self.save_data, f'{self.id}_{idx}.pt'))

        return data

    def __len__(self):
        return self.file.num_row_groups

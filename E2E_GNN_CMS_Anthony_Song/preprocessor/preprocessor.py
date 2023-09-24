import pyarrow.parquet as pq
import numpy as np
import torch 
import torch_geometric
import glob
import os
import pickle
import random
import pandas as pd
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm.auto import tqdm

from base.base_preprocessor import BasePreprocessor
from preprocessor.point_cloud_dataset import PointCloudFromParquetDataset

class MNISTPreprocessor(BasePreprocessor):
    def __init__(self, use, data_dir):
        super(MNISTPreprocessor, self).__init__(use, data_dir)

class TopGunPreprocessor(BasePreprocessor):
    def __init__(self, use, data_dir, num_files,test_ratio,val_ratio,transform_flags, mode, scale_histogram,predict_bins,min_mass,max_mass,num_bins,point_fn,use_pe,num_pe_scales,min_threshold,output_mean_scaling,output_mean_value,output_norm_scaling,output_norm_value, construction = 'knn' , k = 20):
        super().__init__(use, data_dir, num_files,test_ratio,val_ratio,transform_flags, mode, scale_histogram,predict_bins,min_mass,max_mass,num_bins,point_fn,use_pe,num_pe_scales,min_threshold,output_mean_scaling,output_mean_value,output_norm_scaling,output_norm_value,  construction, k)

    def preprocess(self,  num_files,test_ratio,val_ratio,transform_flags,mode, scale_histogram,predict_bins,min_mass,max_mass,num_bins,point_fn,use_pe,num_pe_scales,min_threshold,output_mean_scaling,output_mean_value,output_norm_scaling,output_norm_value,  construction , k ):
        paths = list(glob.glob(os.path.join(self.data_dir, "raw", "*.parquet")))

        dsets = []
        for it,path in enumerate(tqdm(paths[0:num_files])):
            dsets.append(
                PointCloudFromParquetDataset(
                    data_dir = self.data_dir,
                    save_data = os.path.join(self.data_dir, "saved"),
                    id = it,
                    filename = path,
                    transform_flags = transform_flags,
                    mode = mode,
                    point_fn=point_fn,
                    scale_histogram=scale_histogram,
                    predict_bins=predict_bins, min_mass=min_mass, max_mass=max_mass, num_bins=num_bins,
                    use_pe=use_pe, pe_scales=num_pe_scales,
                    suppresion_thresh=min_threshold,
                    output_mean_scaling=output_mean_scaling, output_mean_value=output_mean_value,
                    output_norm_scaling=output_norm_scaling, output_norm_value=output_norm_value,
                    construction =  construction, 
                    k = k
                )
            )

        combined_dset = torch.utils.data.ConcatDataset(dsets)

        sampled_data_size = int(len(combined_dset) * 0.005)

        random_indices = random.sample(range(len(combined_dset)), sampled_data_size)

        combined_dset = torch.utils.data.Subset(combined_dset, random_indices)

        val_size = int(len(combined_dset) * val_ratio)
        test_size = int(len(combined_dset) * test_ratio)
        train_size = len(combined_dset) - val_size - test_size

        train_dset, val_dset, test_dset = torch.utils.data.random_split(
            combined_dset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = train_dset
        self.val_dataset = val_dset
        self.test_dataset = test_dset

class QGPreprocessor(BasePreprocessor):
    def __init__(self, use, data_dir, num_files,test_ratio,val_ratio, construction = 'knn' , k = 15):
        super().__init__(use, data_dir, num_files,test_ratio,val_ratio)

    def preprocess(self,  num_files,test_ratio,val_ratio,  construction = 'knn' , k = 15):

        tau_dataset = list(glob.glob(os.path.join(self.data_dir, "raw", "*.parquet")))
        X_jets = []
        lables = []

        for file_name in tqdm(tau_dataset):
            df = pq.read_table(file_name).to_pandas()
            # X_jets.append(np.array(df['X_jets'].tolist()).astype(np.float32))
            X_jets.append(np.array(df['X_jets'].tolist()))
            lables.append(df['y'].to_numpy())
            # del df
            break
        


        # Get the total number of samples
        num_samples = len(df)

        # Initialize empty arrays for X and y
        X = np.empty((num_samples, 3, 125, 125),dtype=np.float32)
        y = np.empty(num_samples, dtype=int)

        # Iterate through the DataFrame and fill X and y
        for i, row in df.iterrows():
            X[i] =  np.transpose(np.dstack((np.stack(row['X_jets'][0]),np.stack(row['X_jets'][1]),np.stack(row['X_jets'][2]))), (2, 0, 1))
            y[i] = row['y']

        # Rearrange the dimensions of X to match the TensorFlow format (samples, height, width, channels)
        X = np.transpose(X, (0, 2, 3, 1))

        X_data = X.reshape(num_samples, 125*125, 3)
        lables = y


        non_black_pixels_mask = np.any(X_data != 0., axis=-1)

        node_list = []
        for i, x in enumerate(X_data):
            node_list.append(x[non_black_pixels_mask[i]])



        dataset = []
        for i,nodes in enumerate(tqdm(node_list)):
            if construction == 'knn':
                edges = kneighbors_graph(nodes, k, mode='connectivity', include_self=True)
            if construction == 'fc':
                edges = radius_neighbors_graph(nodes, radius= 2e32-1, mode = 'distance', include_self=True)
            if construction == 'epsilon':
                edges = radius_neighbors_graph(nodes, radius= 50, mode = 'distance', include_self=True)

            c = edges.tocoo()
            edge_list = torch.from_numpy(np.vstack((c.row, c.col))).type(torch.long)
            edge_weight = torch.from_numpy(c.data.reshape(-1,1))
            y = np.array(lables[i])
            dataset.append(Data(x=torch.from_numpy(nodes), edge_index=edge_list, edge_attr=edge_weight, y=torch.from_numpy(y).long()))


        rand_seed = 42
        X_train, X_test = train_test_split(dataset, test_size=test_ratio, random_state = rand_seed)
        X_train, X_val = train_test_split(X_train, test_size=val_ratio / (1-test_ratio), random_state = rand_seed)

        self.train_dataset = X_train
        self.val_dataset = X_val
        self.test_dataset = X_test

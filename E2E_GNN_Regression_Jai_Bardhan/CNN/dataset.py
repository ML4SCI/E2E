import pyarrow.parquet as pq
import numpy as np
import torch
import torchvision.transforms as T
import glob
from tqdm.auto import tqdm
import os
from dataset_utils import positional_encoding, zero_suppression

hcal_scale  = 1
ecal_scale  = 0.1
pt_scale    = 0.01
dz_scale    = 0.05
d0_scale    = 0.1
m0_scale    = 85
m1_scale    = 415
p0_scale = 400
p1_scale = 600

class ImageDatasetFromParquet(torch.utils.data.Dataset):
    '''
        Dataset to extract Image from the Parquet File. This does not load
        the entire Parquet Dataset on memory but reads from it every time
        an image is queried.
    '''
    def __init__(
        self,
        filename,
        scale_as_histogram=False,
        transforms=[],
        use_pe=False,
        pe_scales=None,
        use_zero_supression=False,
        min_threshold=None,
        output_mean_scaling=False,
        output_mean_value=None,
        output_norm_scaling=False,
        output_norm_value=None,
    ) -> None:
        '''
            Init fn. of the dataset
            Args:
                filename: The path to the parquet file
                transforms: Transforms to be applied on the image
                use_pe: Whether to use sin/cos positional encoding on the features
                pe_scales: The scales for the positional encoding
                use_zero_suppression: Whether to use zero suppression on the images
                min_threshold: The minimum threshold for zero suppression
                output_mean_scaling: Whether to subtract the ground truth with a mean value
                output_mean_value: The mean value to subtract the ground truth with
                output_norm_scaling: Whether to scale the ground truth with a norm value
                output_norm_value: The norm value to scale the ground truth with

            Returns:
                None
        '''
        super().__init__()

        self.file = pq.ParquetFile(filename)

        self.transforms = T.Compose([T.ToTensor(), *transforms])
        self.use_pe = use_pe
        self.scale_as_histogram = scale_as_histogram
        self.pe_scales = pe_scales
        self.use_zero_suppression = use_zero_supression
        self.min_threshold = min_threshold
        self.output_mean_scaling = output_mean_scaling
        self.output_mean_value = output_mean_value
        self.output_norm_scaling = output_norm_scaling
        self.output_norm_value = output_norm_value

    def __getitem__(
        self,
        idx,
    ):
        '''
            __getitem__ function of a Pytorch dataset. 
            Returns the traning element. 
        '''
        row = self.file.read_row_group(idx).to_pydict()
        to_return = {
            "X_jets":
                self.transforms(np.array(row["X_jet"][0]).transpose(2, 0, 1)).float() if (not self.use_zero_suppression and not self.scale_as_histogram)
                else self.transforms(zero_suppression(np.array(row["X_jet"][0]).transpose(2, 0 , 1), self.min_threshold)).float(),
            "m": torch.as_tensor(row["m"][0], dtype=torch.float),
            "pt": torch.as_tensor(row["pt"][0], dtype=torch.float).unsqueeze(-1),
            "ieta": torch.as_tensor(row["ieta"][0], dtype=torch.float).unsqueeze(-1),
            "iphi": torch.as_tensor(row["iphi"][0], dtype=torch.float).unsqueeze(-1),
        }

        if self.scale_as_histogram:
            to_return['X_jets'][0] *= pt_scale
            to_return['X_jets'][1] *= dz_scale
            to_return['X_jets'][2] *= d0_scale
            to_return['X_jets'][3] *= ecal_scale
            to_return['X_jets'][4] *= hcal_scale
            to_return['pt'] = (to_return['pt'] - p0_scale)/p1_scale 
            to_return['m'] = (to_return['m'] - m0_scale) if not self.output_mean_scaling else to_return['m']
            to_return['m'] = to_return['m'] / m1_scale if not self.output_norm_scaling else to_return['m']
            to_return['iphi'] = to_return['iphi'] / 360.
            to_return['ieta'] = to_return['ieta'] / 140.

            # High value suppression
            to_return['X_jets'][1][to_return['X_jets'][1] < -1] = 0 #(20 cm)
            to_return['X_jets'][1][to_return['X_jets'][1] > 1] = 0 #(20 cm)
            to_return['X_jets'][2][to_return['X_jets'][2] < -1] = 0 #(20 cm)
            to_return['X_jets'][2][to_return['X_jets'][2] > 1] = 0 #(20 cm)

            # Zero suppression
            to_return['X_jets'][0][to_return['X_jets'][0] < 1.e-2] = 0. #(1 GeV)
            to_return['X_jets'][3][to_return['X_jets'][3] < 1.e-2] = 0. #(0.1 GeV)
            to_return['X_jets'][4][to_return['X_jets'][4] < 1.e-2] = 0. #(0.01 GeV)


        if self.use_pe:
            for k in to_return:
                if k != 'm':
                    to_return[k] = positional_encoding(
                        to_return[k], self.pe_scales)

        if self.output_mean_scaling:
            to_return['m'] = to_return['m'] - self.output_mean_value

        if self.output_norm_scaling:
            to_return['m'] = to_return['m'] / self.output_norm_value

        return to_return

    def __len__(self):
        return self.file.num_row_groups


def get_datasets(
    root_dir,
    num_files,
    test_ratio,
    val_ratio,
    scale_as_histogram=False,
    required_transform=None,
    use_pe=False,
    pe_scales=0,
    use_zero_suppression=False,
    min_threshold=0.,
    output_mean_scaling=False,
    output_mean_value=0,
    output_norm_scaling=False,
    output_norm_value=1.,
):
    '''
        Returns the datasets provided the root directory of the multiple parquet files.
        Args:
            root_dir: The root directory containing all the parquet files
            num_files: The number of files to be read
            test_ratio: The ratio of the dataset to be used as the test dataset
            val_ratio: The ratio of the dataset to be used as the validation dataset
            required_transforms: The required transform for the training and validation dataset
            use_pe: Whether to use sin/cos positional encoding
            pe_scales: The scales for the positional encoding
            use_zero_suppression: Whether to use zero suppression on the images
            min_threshold: The minimum threshold for the zero suppression
            output_mean_scaling: Whether to subtract the ground truth with a mean value
            output_mean_value: The mean value to subtract from the ground truth
            output_norm_scaling: Whether to scale the ground truth with a norm value
            output_norm_value: The value with which to scale the ground truth

        Returns:
            train_dset: The training dataset
            val_dset: The validation dataset
            test_dset: The test dataset
            train_size: The size of the training dataset
            val_size: The size of the validation dataset
            test_size: The size of the test dataset
    '''
    paths = list(glob.glob(os.path.join(root_dir, "*.parquet")))

    dsets = []
    for path in tqdm(paths[0:num_files]):
        dsets.append(
            ImageDatasetFromParquet(
                path,
                scale_as_histogram=scale_as_histogram,
                transforms=required_transform,
                use_pe=use_pe, pe_scales=pe_scales,
                use_zero_supression=use_zero_suppression, min_threshold=min_threshold,
                output_mean_scaling=output_mean_scaling, output_mean_value=output_mean_value,
                output_norm_scaling=output_norm_scaling, output_norm_value=output_norm_value
            )
        )

    combined_dset = torch.utils.data.ConcatDataset(dsets)

    val_size = int(len(combined_dset) * val_ratio)
    test_size = int(len(combined_dset) * test_ratio)
    train_size = len(combined_dset) - val_size - test_size

    train_dset, val_dset, test_dset = torch.utils.data.random_split(
        combined_dset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    # test_dset.required_transforms = [T.Normalize(mean=[0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754],
    #                                              std=[10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])]

    return train_dset, val_dset, test_dset, train_size, val_size, test_size

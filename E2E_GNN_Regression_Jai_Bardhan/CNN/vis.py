import glob
import os
from dataset import ImageDatasetFromParquet
from tqdm.auto import tqdm
import torch
import torchvision
import torchvision.transforms as T

root_dir = './data'

paths = list(glob.glob(os.path.join(root_dir, "*.parquet")))

dsets = []
for path in tqdm(paths[0:1]):
    dsets.append(ImageDatasetFromParquet(
        path, transforms=[
            T.Normalize(mean=[0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754],
            std=[10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])

        ]))

combined_dset = torch.utils.data.ConcatDataset(dsets)

image = combined_dset[0]['X_jets']
torchvision.utils.save_image(image[0], 'img0.png')
torchvision.utils.save_image(image[1], 'img1.png')
torchvision.utils.save_image(image[2], 'img2.png')
torchvision.utils.save_image(image[3], 'img3.png')
torchvision.utils.save_image(image[4], 'img4.png')
torchvision.utils.save_image(image[5], 'img5.png')
torchvision.utils.save_image(image[6], 'img6.png')
torchvision.utils.save_image(image[7], 'img7.png')
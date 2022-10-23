import glob
import os
from re import A
from dataset import ImageDatasetFromParquet
from tqdm.auto import tqdm
import torch
from train_utils import AverageMeter

root_dir = './data'

paths = list(glob.glob(os.path.join(root_dir, "*.parquet")))

dsets = []
for path in tqdm(paths[0:1]):
    dsets.append(ImageDatasetFromParquet(
        path, transforms=[]))

combined_dset = torch.utils.data.ConcatDataset(dsets)
loader = torch.utils.data.DataLoader(
    combined_dset, shuffle=True, batch_size=128)

avg_meter = [AverageMeter()] * 8
sq_avg_meter = [AverageMeter()] * 8
rel_count = 0

pbar = tqdm(loader, total=len(loader))
for it, data in enumerate(pbar):
    for i in range(8):
        img = data['X_jets'][:, i, :, :]
        mean = img[img.nonzero(as_tuple=True)].mean()
        count = torch.count_nonzero(img)

        avg_meter[i].update(
            mean, count
        )

        sq_mean = (img[img.nonzero(as_tuple=True)]**2).mean()

        sq_avg_meter[i].update(
            sq_mean, count
        )

        if (it + 1) % 5000 == 0:
            print(avg_meter[i].avg, i)

for i in range(8):
    print(f"Average {i}", avg_meter[i].avg)
    print(f"Std {i}", torch.sqrt(sq_avg_meter[i].avg - avg_meter[i].avg**2))

dev_avg_meter = [AverageMeter()] * 8

pbar = tqdm(loader, total=len(loader))
for data in pbar:
    for i in range(8):
        img = data['X_jets'][:, i, :, :]
        dev = (img[img.nonzero(as_tuple=True)] -
               avg_meter[i].avg.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) ** 2
        count = torch.count_nonzero(img)

        dev_avg_meter[i].update(
            dev.mean(), count
        )

for i in range(8):
    print(f"Std {i}", torch.sqrt(dev_avg_meter[i].avg))

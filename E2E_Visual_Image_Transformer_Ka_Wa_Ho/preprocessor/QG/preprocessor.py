import numpy as np
import pyarrow.parquet as pq

from tqdm import tqdm
from numba import njit

import torch
import math
import os

big_tensor = []
big_target = []

def generate(pf, mean_std, max_min):
    sum_channel, sum2_channel, size = np.array([0,0,0]), np.array([0,0,0]), 0

    batch_size = 4*1024
    nchunks = math.ceil(pf.num_row_groups/batch_size)
    record_batch = pf.iter_batches(batch_size=batch_size)
    for which_batch in tqdm(range(nchunks)):
        batch = next(record_batch)

        sum_channel_tmp, sum2_channel_tmp, size_tmp = process(batch, mean_std, max_min)

        sum_channel+=sum_channel_tmp
        sum2_channel+=sum2_channel_tmp
        size+=size_tmp

    return sum_channel, sum2_channel, size

def process(batch, mean_std):
    p = batch.to_pandas()
    im = np.array(np.array(np.array(p.iloc[:, 0].tolist()).tolist()).tolist())
    meta = np.array(p.iloc[:, 3]).astype(np.bool_)
    return saver(im, mean_std)


@njit
def jit_helper(im, mean_channels, std_channels, use_mean_std):
    B, C, H, W = im.shape
    sum_channel, sum2_channel = np.array([0, 0, 0]), np.array([0, 0, 0])
    
    for b in range(B):
        for c in range(3):
            for h in range(125):
                im[b,c,h,:][im[b,c,h,:] < 1.e-3] = 0

    #Normalize
    for c in range(3):
      if use_mean_std:
        mean, std = mean_channels[c], std_channels[c]
      else:
        sum_channel[c], sum2_channel[c] = np.sum(im[:,c,:,:]), np.sum(im[:,c,:,:]**2)
        mean, std = sum_channel[c]/(B*H*W), np.sqrt(sum2_channel[c]/(B*H*W) -  (sum_channel[c]/(B*H*W))**2)

      im[:,c,:,:] = (im[:,c,:,:] - mean)/std
      for b in range(B):
          #Scale to [0,255] and save as png
          im[b,c,:,:] *= 255/im[b,c,:,:].max()
          batch_std = im[b,c,:,:].std()
          for h in range(125):
              #Clip values < 0 or > 500 sigmas
              im[b,c,h,:][(im[b,c,h,:]<0)|(im[b,c,h,:]>500*batch_std)] = 0

    im = np.transpose(im, (0,2,3,1))
    im = im.astype(np.uint8)
    
    return im, sum_channel, sum2_channel, B*H*W

def saver(im, mean_std):
    if not (mean_std[0] is None):
      use_mean_std = True
      mean, std = mean_std
    else:
      use_mean_std = False
      mean, std = np.array([0,0,0]), np.array([0,0,0])
    im, sum_channel, sum2_channel, size = jit_helper(im, mean, std, use_mean_std)
   
    global big_tensor
    global big_target

    big_tensor.append(im)
    big_target.append(meta)

    return sum_channel, sum2_channel, size

def runner(source, mean_std=(None,None)):

    sumw, sumw2, size = np.array([0,0,0]), np.array([0,0,0]), 0

    files = os.listdir(source)

    print("The following files were found in the provided Directory")
    print(files)

    for i in range(len(files)):
        print(f'Running {files[i]}')
        sum_tmp, sum2_tmp, size_tmp = generate(pq.ParquetFile(os.path.join(source, files[i])), mean_std=mean_std)

        sumw+=sum_tmp
        sumw2+=sum2_tmp 
        size+=size_tmp

    mean, std = sumw/size, np.sqrt( sumw2/size - (sumw/size)**2 )
    print('mean and std are ', mean, std)
    print("The files were successfully generated")
    return mean, std

if __name__=='__main__':

    os.makedirs("data/QG/tensor", exist_ok=True)

    mean, std = runner(source="data/QG/parquet/train")
    
    big_tensor = torch.from_numpy(np.concatenate(big_tensor))
    big_target = torch.from_numpy(np.concatenate(big_target))

    torch.save({'image':big_tensor, 'target':big_target}, "data/QG/tensor/train.pt")

    print(f'Tensor shape is {big_tensor.shape}')
    print(f'Target shape is {big_target.shape}')

    big_tensor = []
    big_target = []

    runner(source="data/QG/parquet/test", mean_std=(mean, std))

    big_tensor = torch.from_numpy(np.concatenate(big_tensor))
    big_target = torch.from_numpy(np.concatenate(big_target))

    torch.save({'image':big_tensor, 'target':big_target}, "data/QG/tensor/test.pt")

    print(f'Tensor shape is {big_tensor.shape}')
    print(f'Target shape is {big_target.shape}')

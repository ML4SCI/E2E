import numpy as np
import pyarrow.parquet as pq

import math
import os
import glob
import tensorflow as tf

def _image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def tf_helper(im, meta, writer, which_file, which_batch, mean_channels, std_channels, use_mean_std):
    B, C, H, W = im.shape
    sum_channel, sum2_channel, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0
    
    im[im < 1.e-3] = 0

    #Normalize
    for c in range(8):
      if use_mean_std:
        mean, std = mean_channels[c], std_channels[c]
      else:
        sum_channel[c], sum2_channel[c] = np.sum(im[:,:,:,c]), np.sum(im[:,:,:,c]**2)
        mean, std = sum_channel[c]/(B*H*W), np.sqrt(sum2_channel[c]/(B*H*W) -  (sum_channel[c]/(B*H*W))**2)

      im[:,c,:,:] = (im[:,c,:,:] - mean)/std

      #Scale to [0,255] and save as png
      max_ = im[:,:,:,c].max(axis=(1,2), keepdims=True)
      max_[max_==0] = 1
      im[:,:,:,c] *= 255/max_
      batch_std = im[:,:,:,c].std(axis=(1,2), keepdims=True)
      #Clip values < 0 or > 500 sigmas
      im[:,:,:,c][(im[:,:,:,c]<0)|(im[:,:,:,c]>500*batch_std)] = 0
    for _b in range(B):
      data = {} 
      data['name'] = _bytes_feature(f'{which_file}_{which_batch}_{_b}')
      data['label'] = _int64_feature(meta[_b])
      for c in range(8):
        data[f'ch{c}'] =  _image_feature(im[_b,:,:,c][:,:,None].astype(np.uint8))

      out = tf.train.Example(features=tf.train.Features(feature=data))
      writer.write(out.SerializeToString()) 

    return sum_channel, sum2_channel, B*H*W

def saver(path, im, meta, writer, which_file, which_batch, mean_std):

    if not (mean_std[0] is None):
      use_mean_std = True
      mean, std = mean_std
    else:
      use_mean_std = False
      mean, std = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0])

    sum_channel, sum2_channel, size = tf_helper(im, meta, writer, which_file, which_batch, mean, std, use_mean_std)

    return sum_channel, sum2_channel, size

def process(batch, path, writer, which_file, which_batch, mean_std):
    p = batch.to_pandas()
    im = np.array(p.iloc[:,0].tolist()).reshape((-1,125,125,8))
    meta = np.array(p.iloc[:,1]).astype(np.intc)
    return saver(path, im, meta, writer, which_file, which_batch, mean_std)

def generate(pf, path, writer, which_file, mean_std):
    sum_channel, sum2_channel, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0

    batch_size = pf.num_row_groups
    nchunks = math.ceil(pf.num_row_groups/batch_size)
    record_batch = pf.iter_batches(batch_size=batch_size)
    for which_batch in range(nchunks):
        batch = next(record_batch)
        sum_channel_tmp, sum2_channel_tmp, size_tmp = process(batch, path, writer, which_file, which_batch, mean_std)
        sum_channel+=sum_channel_tmp
        sum2_channel+=sum2_channel_tmp
        size+=size_tmp
    return sum_channel, sum2_channel, size

def runner(source, indexes, target, mean_std=(None,None), is_valid=False):
    """
    Fuction to convert all the Parquet Files in a given folder to .png format Files
    Args:
    source: The souce folder of the Parquet Files
    target: The target folder where the dataset will be stored
    """

    sumw, sumw2, size = np.array([0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0]), 0

    files = [source+f'/BoostedTop_x1_fixed_{i}.snappy.parquet' for i in indexes]

    print("The following files will be processed")
    print(files)

    if mean_std[0] is None:
      if is_valid:
        sub_str = 'valid'
      else:
        sub_str = 'train'
    else: 
      sub_str = 'test'

    writer = tf.io.TFRecordWriter(f"Top_TFRecords_shard_{sub_str}.tfrecords")
    for i in range(len(files)):
        try:
            sum_tmp, sum2_tmp, size_tmp = generate(pq.ParquetFile(files[i]), target, writer, which_file=indexes[i], mean_std=mean_std)
  
            sumw+=sum_tmp
            sumw2+=sum2_tmp 
            size+=size_tmp
        except:
            continue

    writer.close() 
    mean, std = sumw/size, np.sqrt( sumw2/size - (sumw/size)**2 )

    print("The files were successfully generated")
    return sumw, sumw2, size

if __name__=='__main__':
    
    os.makedirs("data/top/tf", exist_ok=True)

    sumw_t, sumw2_t, size_t  = runner(source="data/top/parquet", indexes=range(814), target="data/top/tf")
    sumw_v, sumw2_v, size_v  = runner(source="data/top/parquet", indexes=range(814,1002), target="data/top/tf", is_valid=True)
  
    sumw = sumw_t+sumw_v
    sumw2 = sumw2_t+sumw2_v
    size = size_t+size_v
    mean, std = sumw/size, np.sqrt( sumw2/size - (sumw/size)**2 )
    runner(source="data/top/parquet", indexes=range(1003,1250), target="data/top/tf", mean_std=(mean, std))

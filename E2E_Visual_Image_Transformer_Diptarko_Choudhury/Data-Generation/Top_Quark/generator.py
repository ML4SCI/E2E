import tensorflow as tf
import numpy as np
import os
import random
from tqdm.auto import tqdm
import cv2
import pyarrow.parquet as pq
import glob
import math
import cupy as cp

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

def parser(image,label,name):

    data = {
        'channel_1' : _image_feature((image[:,:,0])[:,:,None]),
        'channel_2' : _image_feature((image[:,:,1])[:,:,None]),
        'channel_3' : _image_feature((image[:,:,2])[:,:,None]),
        'channel_4' : _image_feature((image[:,:,3])[:,:,None]),
        'channel_5' : _image_feature((image[:,:,4])[:,:,None]),
        'channel_6' : _image_feature((image[:,:,5])[:,:,None]),
        'channel_7' : _image_feature((image[:,:,6])[:,:,None]),
        'channel_8' : _image_feature((image[:,:,7])[:,:,None]),
        'label' : _int64_feature(label),
        'name' : _bytes_feature(name)
    }
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


BATCH_SIZE = 320  # This is the batch size for reading the parquet files
MULTIPLE = 4  # Number of ParquetFiles that will be contained in each TFRecord File

def generate(pf, writer, ab):
    record_batch = pf.iter_batches(batch_size=BATCH_SIZE)
    while True:
        try:

            batch = next(record_batch)
            ab = transform(batch, writer, ab)

        except StopIteration:
            # print("Done")
            return ab

def transform(batch, writer, ab):
    p = batch.to_pandas()
    im = cp.array(p.iloc[:,0].tolist())
    meta = np.array(p.iloc[:,1])
    return saver(im,meta, writer, ab)


def saver(im, meta, writer, ab):
    alpha, beta = ab

    im[im < 1.e-3] = 0 #Zero_suppression
    im = cp.reshape(im, (BATCH_SIZE,125,125,8))
    for _z in range(8):
        im[:,:,:,_z] = (im[:,:,:,_z] - im[:,:,:,_z].mean())/(im[:,:,:,_z].std())
        im[:,:,:,_z] = cp.clip(im[:,:,:,_z], a_min = 0, a_max = 500*im[:,:,:,_z].std(axis = (1,2))[:,None,None])

    im = im.get()
    for i in range(meta.shape[0]):
        img = im[i,:,:,:]
        new_img = np.random.randn(128,128,8)
        for _z in range(8):
            # img[:,:,_z] = np.clip(img[:,:,_z], 0, 500*img[:,:,_z].std())
            if (img[:,:,_z]==0).all() == False:
                img[:,:,_z] = 255*(img[:,:,_z])/(img[:,:,_z].max())

            new_img[:,:,_z] = cv2.resize(img[:,:,_z],(128,128))

            if (new_img[:,:,_z]==0).all() == False:
                new_img[:,:,_z] = 255*(new_img[:,:,_z])/(new_img[:,:,_z].max())

        img = new_img.astype(np.uint8)

        if(meta[i] == 0):
            name = "0_"+str(alpha)
            label = 0
            alpha = alpha + 1

        if(meta[i] == 1):
            name = "1_"+str(beta)
            label = 1
            beta = beta + 1

        item = parser(img, label, name)
        writer.write(item.SerializeToString())

    return [alpha, beta]


def runner(source, target):

    """
    Function to convert Parquet Files into TFRecord Files
    Args:
    source: The source folder where the ParquetFiles are located
    target: The target folder where the TFRecord Files will be created

    """
    ab = [0, 0]
    files = glob.glob(os.path.join(source,"*.parquet"))
    print("Got the following files in the source folder")
    print(files)

    no_tfr = math.ceil(len(files)/MULTIPLE)
    random.shuffle(files)
    j = 0

    for i in tqdm(range(len(files))):
        if i%multiple==0:
            j = j+1
            writer = tf.io.TFRecordWriter(os.path.join(target,f"Top_TFRecords_shard_{j}_of_{no_tfr}.tfrecords"))
        ab = generate(pq.ParquetFile(files[i]), writer, ab)

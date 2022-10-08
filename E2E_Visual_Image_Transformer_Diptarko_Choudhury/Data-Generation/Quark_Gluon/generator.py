from tkinter.tix import MAIN
import numpy as np
from pip import main
path_test = "Test/"
path_train = "Train/"
import cv2
import pyarrow.parquet as pq
import gc
import time as time
import os


def generate(pf, path, ab):
    record_batch = pf.iter_batches(batch_size=4*1024)
    while True:
        try:
            batch = next(record_batch)
            ab = transform(batch, path, ab)

        except StopIteration:
            return ab


def transform(batch, path, ab):
    p = batch.to_pandas()
    im = np.array(np.array(np.array(p.iloc[:, 0].tolist()).tolist()).tolist())
    meta = np.array(p.iloc[:, 3])
    return saver(im, meta, path, ab)


def saver(im, meta, path, ab):
    alpha, beta = ab

    im[im < 1.e-3] = 0 #Zero_suppression
    im[:,0,:,:] = (im[:,0,:,:] - im[:,0,:,:].mean())/(im[:,0,:,:].std())
    im[:,1,:,:] = (im[:,1,:,:] - im[:,1,:,:].mean())/(im[:,1,:,:].std())
    im[:,2,:,:] = (im[:,2,:,:] - im[:,2,:,:].mean())/(im[:,2,:,:].std())

    for i in range(meta.shape[0]):
        img = im[i,:,:,:]

        channel1 = img[0,:,:]
        channel2 = img[1,:,:]
        channel3 = img[2,:,:]

        channel1 = np.clip(channel1, 0, 500*channel1.std())
        channel2 = np.clip(channel2, 0, 500*channel2.std())
        channel3 = np.clip(channel3, 0, 500*channel3.std())

        channel1 = 255*(channel1)/(channel1.max())
        channel2 = 255*(channel2)/(channel2.max())
        channel3 = 255*(channel3)/(channel3.max())

        img[0,:,:] = channel1
        img[1,:,:] = channel2
        img[2,:,:] = channel3

        img = img.astype(np.uint8)
        img = img.T

        if(meta[i] == 0):
            impath = os.path.join(path,"0",str(str(alpha)+".png"))
            alpha = alpha + 1
        if(meta[i] == 1):
            impath = os.path.join(path,"1",str(str(beta)+".png"))
            beta = beta + 1

        cv2.imwrite(impath , img)

    return [alpha, beta]


def runner(source, target):
    """
    Fuction to convert all the Parquet Files in a given folder to .png format Files
    Args:
    source: The souce folder of the Parquet Files
    target: The target folder where the dataset will be stored
    """
    ab = [0, 0]

    os.mkdir(os.path.join(target,"1"))
    os.mkdir(os.path.join(target,"0"))
    files = os.listdir(source)
    print("The following files were found in the provided Directory")
    print(files)
    for i in range(len(files)):
        ab = generate(pq.ParquetFile(os.path.join(source, files[i])), target, ab)
    print("The files were successfully generated")

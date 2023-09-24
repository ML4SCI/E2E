import torch
import wandb
import gc
import sys
import time
import os
import math
import argparse
import random

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from PIL import Image

sys.path.append('models/')

from models import *

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, TensorDataset 

from torchvision import datasets, transforms

from tqdm.auto import tqdm

from sklearn import metrics

parser = argparse.ArgumentParser(description='Script to run training of ViT models')
parser.add_argument('-d', '--dataset', type=str, default='QG', help='dataset to train (default: %(default)s)')
parser.add_argument('-gpu', '--gpu', type=int, default=2, help='number of gpu to use (default: %(default)s)')
parser.add_argument('-p', '--patch', type=int, default=2, help='patch size (default: %(default)s)')
parser.add_argument('-w', '--win', type=str, default='4', help='window size separated by commas (default: %(default)s)')
parser.add_argument('-e', '--emb', type=int, default=48, help='embedding dimension (default: %(default)s)')
parser.add_argument('-h', '--head', type=str, default='3,6,12,24', help='number of heads in the multi-head attention in the different layers separated by commas (default: %(default)s)')
parser.add_argument('-sw', action='store_true', help='use shifting window operation as in Swin')
parser.add_argument('-cw', action='store_true', help='use layer-wise convolution as in Win')
args = parser.parse_args()

args.win = [int(_) for _ in args.win.split(',')]
if len(args.win)==1: args.win = args.win[0]
args.head = (int(_) for _ in args.head.split(','))

use_gpu = torch.cuda.is_available()

max_test_auc = [-999, -999]
best_val_auc = [-999, 0]
 
init_time = time.time()

import timm 
import re

from timm.layers import Mlp
from collections import OrderedDict

from timm.layers import ClassifierHead
from swin_based_transformer import SwinTransformer, _create_swin_transformer

def swin_based_model(pretrained=False, **kwargs):
    model_args = dict(img_size=128, patch_size=args.patch, window_size=args.win, embed_dim=args.emb, depths=(2, 2, 6, 2), num_heads=args.head, shift_win=args.sw, conv_win=args.cw)
    return _create_swin_transformer(
        'swin_based_model', pretrained=pretrained, **dict(model_args, **kwargs))

config = {
    "dataset": args.dataset,
    "model": swin_based_model,
    "log_intr": 100,
    "n_workers": 0,
    "ngpu": args.gpu,
    "batch_size": 96,
    "optimizer": [torch.optim.AdamW, {'lr': 0.0001, 'weight_decay': 0.05}], 
    "scheduler": [torch.optim.lr_scheduler.ReduceLROnPlateau, {'threshold': 0.001, 'patience': 3, 'factor': 0.5}],
    "loss": F.binary_cross_entropy_with_logits,
    "aug": [transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.04,0))]
}
  
class CustomToTensor(object):
    def __call__(self, img):
        pic = torch.from_numpy(np.array(img))
        pic = pic.reshape((125,125,9))[:,:,:8].contiguous()
        return pic.permute(2,0,1).div(255)

if config['aug']:
    train_transform = transforms.Compose(config['aug'])

if config['dataset'] == 'QG':
    common_transform = transforms.Compose([transforms.Pad([2,2,1,1])])
elif config['dataset'] == 'top':
    common_transform = transforms.Compose([CustomToTensor(), transforms.Pad([2,2,1,1])])
else:
    print('Dataset does not exist')
    raise 

def setup(rank, world_size=config['ngpu']):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29509'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, dataset, isTrain, world_size=config['ngpu'], batch_size=config['batch_size'], pin_memory=False, num_workers=config['n_workers']):
    if isTrain:
        drop_last = False
        shuffle = True 
    else: 
        drop_last = True
        shuffle = False
    #Load training/testing dataset into tensors
    if config['ngpu']==1:
        dl = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle, drop_last=False) 
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=drop_last, shuffle=shuffle)
        dl = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=False, sampler=sampler)
    return dl

def cleanup():
    dist.destroy_process_group()

def runTrain(model, rank, dataloader_train, scaler, optimizer, world_size=config['ngpu']):
  model.train()

  train_loss, auc = torch.tensor([0], dtype=float, requires_grad=False), 0
  if use_gpu: train_loss = train_loss.to(rank)

  total_step, total_loss = 1, 0
  label_list, outputs_list = [], []

  train_gather_object = [None for _ in range(world_size)]

  for step, (image, label) in enumerate(tqdm(dataloader_train)):
      with torch.cuda.amp.autocast() if use_gpu else nullcontext():
          if use_gpu:
              image, label = image.to(rank), label.to(rank)

          if config['dataset'] == 'QG':
              image = common_transform(image/255)

          label = label.float()[:, None]

          if config['aug']:
              image = train_transform(image)

          for param in model.parameters():
              param.grad = None
    
          outputs = model(image)
          loss = config['loss'](outputs, label)

      if use_gpu:
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
      else:
          loss.backward()
          optimizer.step()

      if step%config['log_intr'] == 0:
          wandb.log({"train_loss": loss})

      with torch.no_grad():
          train_loss+=loss
          label_list.append(label)
          outputs_list.append(outputs.flatten().sigmoid())

  with torch.no_grad():
    data = {
          'nstep': len(dataloader_train),
          'loss': train_loss.item(),
          'label_list': torch.cat(label_list).detach().cpu().tolist(),
          'outputs_list': torch.cat(outputs_list).detach().cpu().tolist()
      }

    if(rank == 0):
        dist.gather_object(data, object_gather_list=train_gather_object)
        label_list, outputs_list = [], []
        for items in train_gather_object:
            total_step+=items['nstep']
            total_loss+=items['loss']
            label_list+=items['label_list']
            outputs_list+=items['outputs_list']
        auc = metrics.roc_auc_score(label_list, outputs_list)
    else:
        dist.gather_object(data)

  return auc, total_loss/total_step

@torch.no_grad()
def runEval(model, rank, dataloader, test_val='test', world_size=config['ngpu']):
  model.eval()

  eval_loss, auc = torch.tensor([0], dtype=float, requires_grad=False), 0
  if use_gpu: eval_loss = eval_loss.to(rank)

  total_step, total_loss = 1, 0
  label_list, outputs_list = [], []

  test_gather_object = [None for _ in range(world_size)]

  for step, (image, label) in enumerate(tqdm(dataloader)):

      if use_gpu:
        image, label = image.to(rank), label.to(rank)

      with torch.cuda.amp.autocast() if use_gpu else nullcontext():
          if config['dataset'] == 'QG':
              image = common_transform(image/255)
          label = label.float()[:, None]
          outputs = model(image)
          loss = config['loss'](outputs, label)
          label_list.append(label)
          outputs_list.append(outputs.flatten().sigmoid())
          eval_loss+=loss

  data = {
        'nstep': len(dataloader),
        'loss': eval_loss.item(),
        'label_list': torch.cat(label_list).detach().cpu().tolist(),
        'outputs_list': torch.cat(outputs_list).detach().cpu().tolist()
    }
  
  if(rank == 0):
      dist.gather_object(data, object_gather_list=test_gather_object)
      label_list, outputs_list = [], []
      for items in test_gather_object:
          total_step+=items['nstep']
          total_loss+=items['loss']
          label_list+=items['label_list']
          outputs_list+=items['outputs_list']
      auc = metrics.roc_auc_score(label_list, outputs_list)
  else:
      dist.gather_object(data)

  return auc, total_loss/total_step

def main(rank, dataset_Train, dataset_Valid, dataset_Test, run, world_size=config['ngpu']):
  setup(rank, world_size)

  torch.cuda.set_device(rank)

  dataloader_train = prepare(rank, dataset_Train, True) 
  dataloader_val = prepare(rank, dataset_Valid, False) 
  dataloader_test = prepare(rank, dataset_Test, False) 

  if config['dataset'] == 'QG':
      in_chans = 3
  elif config['dataset'] == 'top':
      in_chans = 8

  net = config['model'](num_classes=1, in_chans=in_chans)
  if use_gpu: 
      net = net.to(rank)
      net = DDP(net, device_ids=[rank], output_device=rank)
  
  optimizer = config['optimizer'][0](net.parameters(), **config['optimizer'][1])
  scheduler = config['scheduler'][0](optimizer, 'max', verbose=True, **config['scheduler'][1])

  run.watch(net, log_freq=config['log_intr'])
  scaler = torch.cuda.amp.GradScaler()
  
  for epoch in range(100):
 
      if config['ngpu'] > 1: dataloader_train.sampler.set_epoch(epoch) 

      train_auc, train_loss = runTrain(net, rank, dataloader_train, scaler, optimizer)
      test_auc, test_loss = runEval(net, rank, dataloader_test, 'test')
      val_auc, val_loss = runEval(net, rank, dataloader_val, 'val')
  
      group = dist.new_group(list(range(world_size)))

      if rank==0:
          torch.save({
                  'epoch': epoch,
                  'model_state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()
                  }, f"model_epoch{epoch}.pt")
          run.log({
                  "Epoch": epoch,
                  "time": time.time()-init_time,
                  "Train_auc_epoch": train_auc,
                  "Test_auc_epoch": test_auc,
                  "Val_auc_epoch": val_auc,
                  "Train_loss_epoch": train_loss,
                  "Test_loss_epoch": test_loss,
                  "Val_loss_epoch": val_loss,
                  })

          global max_test_auc
          global best_val_auc

          #Keep track of best test auc, and corresponding epoch
          if test_auc > max_test_auc[0]: max_test_auc = [test_auc, epoch] 

          #Early stopping
          if val_auc-best_val_auc[0] < 0.0001: 
              best_val_auc[1] += 1
          else:
              best_val_auc[0] = val_auc 
              best_val_auc[1] = 0
          if best_val_auc[1] > 10: 
              val_auc = -1
              print('Early stopping')

          to_broadcast = torch.tensor([val_auc], dtype=torch.float32).to(rank)
      else:
          to_broadcast = torch.empty(1).to(rank)

      dist.broadcast(to_broadcast, src=0, group=group)
      if to_broadcast[0]<0: break

      scheduler.step(to_broadcast[0])
  
      gc.collect()
  cleanup()

if __name__ == '__main__':

    wandb.login()
    run = wandb.init(
          project="Vision_Transformers",
          name="run",
          config=config 
          )
  
    run.define_metric( 
                "Train_auc_epoch",
                summary='max')
    run.define_metric( 
                "Test_auc_epoch",
                summary='max')
    run.define_metric( 
                "Val_auc_epoch",
                summary='max')
    run.define_metric( 
                "Train_loss_epoch",
                summary='min')
    run.define_metric( 
                "Test_loss_epoch",
                summary='min')
    run.define_metric( 
                "Val_loss_epoch",
                summary='min')
  
    if config['dataset'] == 'QG':
        train_tensor = torch.load('data/QG/tensor/train.pt')
        dataset_Train = TensorDataset(train_tensor['image'], train_tensor['target']) 
        print('Finish loading training set')
  
        test_tensor = torch.load('data/QG/tensor/test.pt')
        dataset_Test = TensorDataset(test_tensor['image'], test_tensor['target']) 
        print('Finish loading testing set')
  
    elif config['dataset'] == 'top':
        dataset_Train = datasets.ImageFolder('data/top/train', transform=common_transform)
        dataset_Test  = datasets.ImageFolder('data/top/test', transform=common_transform)
    
    train_valid_samples = len(dataset_Train)
    test_samples = len(dataset_Test)
    
    total_samples = train_valid_samples+test_samples
    
    print(f'There are {train_valid_samples} training samples and {test_samples} testing samples.')
    
    dataset_Train, dataset_Valid = torch.utils.data.random_split(dataset_Train, (train_valid_samples-int(total_samples*0.1), int(total_samples*0.1)), generator=torch.Generator().manual_seed(42))
      
    mp.spawn(
        main,
        args=[dataset_Train, dataset_Valid, dataset_Test, run],
        nprocs=config['ngpu'],
        join=True
    )

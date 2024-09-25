import numpy as np  
import torch
import subprocess
import random
import os
import torch.distributed as dist
from torch.distributed import init_process_group
from pathlib import Path
from torchvision import transforms
from timm.data.transforms_factory import create_transform

QUARK_GLUON_MEAN = [0.00036276, 0.00050321, 0.00560932]
QUARK_GLUON_STD = [0.00023162, 0.00031747, 0.00257909]

BOOSTED_TOP_MEAN = [0.00030618, 0.00032988, 0.00022402, 0.00050185, 0.00590864,
       0.00434622, 0.00457014, 0.00442105]
BOOSTED_TOP_STD = [1.57844912e-04, 1.27444286e-04, 9.95345503e-05, 3.33565491e-04,
       3.09065050e-03, 1.89672124e-03, 2.02093365e-03, 2.01831124e-03]


def get_gauss(s,sig):
    x = np.linspace(0, s, s)
    y = np.linspace(0, s, s)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((x - s/2)**2 + (y - s/2)**2) / (2 * sig**2))
    return z

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir,exist_ok=True)
        sync_file = 'file://%s/pytorch_sync.%s' % (
                sync_file_dir, os.environ['SLURM_JOB_ID'])
        return sync_file



def ddp_setup():
    WORLD_SIZE = int(os.environ['SLURM_NTASKS']) 
    GLOBAL_RANK = int(os.environ['SLURM_PROCID'])
    print(f'WORLD_SIZE: {WORLD_SIZE}, GLOBAL_RANK: {GLOBAL_RANK}')
    sync_file = _get_sync_file()
    init_process_group(backend="nccl",init_method=sync_file,rank=GLOBAL_RANK,world_size=WORLD_SIZE)

def ddp_setup_single_node(args):
    """ Setup for single-node multi-GPU training """
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Default to 0 if LOCAL_RANK isn't set.
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.global_rank = dist.get_rank()


def get_transform(args, mode):
    list_of_transforms = []
    if mode == 'train':
        list_of_transforms.append(transforms.RandomResizedCrop(125, interpolation=3))
        list_of_transforms.append(transforms.RandomHorizontalFlip())
    

    if Path(args.data_dir).parent.stem == 'QG':
        list_of_transforms.append(transforms.Normalize(mean=QUARK_GLUON_MEAN, std=QUARK_GLUON_STD))
    else:
        list_of_transforms.append(transforms.Normalize(mean=BOOSTED_TOP_MEAN, std=BOOSTED_TOP_STD))

# This is only for dist-ft.py
def build_transform(is_train, args):
    if Path(args.data_dir).parent.stem == 'QG':
        mean = QUARK_GLUON_MEAN
        std = QUARK_GLUON_STD
    else:
        mean = BOOSTED_TOP_MEAN
        std = BOOSTED_TOP_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=125,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    return transforms.Normalize(mean, std)
    

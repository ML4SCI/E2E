import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
from einops import rearrange
import mlflow
import argparse
from model import ViTMAE
from ericmodel import ViTMAE as EricViTMAE
from utils import *


def get_args_parser():
    
    # Difference between lr and base lr
    # What is weight decay
    #

    #TODOs:
    # Plan for grid-wise masking 
    # lr scheduler
    # warmup epochs
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str,required=True)
    parser.add_argument('--model', type=str, default='vitmae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mask', type=str, default=0.75)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--decoder_dim', type=int, default=128)
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--enc_heads', type=int, default=12)
    parser.add_argument('--dec_heads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:15')
    parser.add_argument('-cm', '--centre_masking',action='store_true')
    parser.add_argument('-rrc',action='store_true')
    parser.add_argument('-rhf',action='store_true')
    args = parser.parse_args()
    return args



optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def pre_train(args,         
              dataset_train,
              ):
    
    list_of_transforms = []
    if args.rrc:
        list_of_transforms.append(transforms.RandomResizedCrop(125, scale=(0.2, 1.0), interpolation=3))
    if args.rhf:
        list_of_transforms.append(transforms.RandomHorizontalFlip())
    
    list_of_transforms.append(transforms.Normalize(mean=[0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 ,
       1.1099757 , 1.15771   , 1.1159292 ], std=[ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 ,
       12.441227 , 12.12112  , 11.721005 ]))

    transform_train = transforms.Compose(list_of_transforms)
    device = torch.device(f'{args.device}')
    
    dataset_train = torch.Tensor(dataset_train)
    dataset_train = rearrange(dataset_train, 'b h w c-> b c h w')
    if len(list_of_transforms)>0:
        dataset_train = transform_train(dataset_train)
    dataset_train = TensorDataset(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    model = ViTMAE(embed_dim=args.encoder_dim,
                    decoder_embed_dim=args.decoder_dim,
                    depth=args.encoder_depth,
                    decoder_depth=args.decoder_depth,
                    num_heads=args.enc_heads,
                    decoder_num_heads=args.dec_heads,
                    ) if args.model.lower() == 'vitmae' else EricViTMAE(embed_dim=args.encoder_dim,
                    decoder_embed_dim=args.decoder_dim,
                    depth=args.encoder_depth,
                    decoder_depth=args.decoder_depth,
                    num_heads=args.enc_heads,
                    decoder_num_heads=args.dec_heads,
                    )

    # model = nn.DataParallel(model)
    model = model.to(device)


    if args.optim.lower() != 'sgd':
        optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, 20))

    mlflow.set_experiment('Basic Experiment')
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params({
        'runname': args.runname,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'optimizer': optimizer,
        'transform_train': transform_train,
        'mask_ratio': args.mask,
        'optim': args.optim,
        'momentum': args.momentum,
        'decoder_depth': args.decoder_depth,
        'encoder_depth': args.encoder_depth,
        'decoder_dim': args.decoder_dim,
        'encoder_dim': args.encoder_dim,
        'enc_heads': args.enc_heads,
        'dec_heads': args.dec_heads,
        'centered masking': args.cm,
        'rrc': args.rrc,
        'rhf':args.rhf
    })

    Path(f'Pre-Training Experiments/{args.runname}').mkdir(exist_ok=True, parents=True)
    global_min_loss = 1000000
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = 0
        for batch, (data, ) in enumerate(data_loader_train):
            data = data.to(device)
            optimizer.zero_grad()
            loss,_,_ = model(data,mask_ratio=float(args.mask))
            loss = loss.mean()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch / len(data_loader_train))
        epoch_loss /= (batch+1)
        if epoch_loss < global_min_loss:
            global_min_loss = epoch_loss
            torch.save(model.state_dict(), f'Pre-Training Experiments/{args.runname}/best.pt')
        mlflow.log_metric('loss', epoch_loss, step=epoch)
        

    torch.save(model.state_dict(), f'Pre-Training Experiments/{args.runname}/last.pt')
    mlflow.end_run()

if __name__ == '__main__':
    args = get_args_parser()
    data = h5py.File('../Dataset_Specific_Unlabelled.h5', 'r')  
    data = data['jet'][...]
    pre_train(args, data)
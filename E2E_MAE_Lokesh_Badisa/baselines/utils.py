import torch
import torch.nn as nn
import mlflow
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from einops import rearrange
from torchvision import transforms
from baselines.models.resnet import *
from baselines.models.inception import *
from baselines.models.efficientnet import *
from baselines.models.vit import *    
from baselines.models.groupvit import *

transform = transforms.Normalize(mean=[0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 ,
       1.1099757 , 1.15771   , 1.1159292 ], std=[ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 ,
       12.441227 , 12.12112  , 11.721005 ])



modelmap = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "inception": inception_v3,
    "efficientnetb0": efficientnet_b0,
    "efficientnetb1": efficientnet_b1,
    "efficientnetb2": efficientnet_b2,
    "efficientnetb3": efficientnet_b3,
    "efficientnetb4": efficientnet_b4,
    "efficientnetb5": efficientnet_b5,
    "efficientnetb6": efficientnet_b6,
    "efficientnetb7": efficientnet_b7,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l,
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "crossvit_tiny": crossvit_tiny,
    "crossvit_small": crossvit_small,
    "crossvit_base": crossvit_base,
    "groupvit_tiny": groupvit_tiny,
    "groupvit_small": groupvit_small,
    "groupvit_base": groupvit_base,
    "groupvit_large": groupvit_large,
    "groupvit_tiny2": groupvit_tiny2,
    # "swint_tinyw9": swint_tinyw9,
    # "swint_smallw9": swint_smallw9,
    # "swint_basew9": swint_basew9
}

optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_weights(wpath):
    weights = torch.load(wpath)
    for key in list(weights.keys()):
        weights[key.replace('module.','')] = weights.pop(key)
    return weights

def load_data(args):
    data = h5py.File('../Dataset_Specific_labelled_train_test.h5', 'r')
    X_train = data['X_train'][...]
    X_test = data['X_test'][...]
    y_train = data['Y_train'][...]
    y_test = data['Y_test'][...]
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    X_train = rearrange(X_train, 'b h w c -> b c h w')
    X_test = rearrange(X_test, 'b h w c -> b c h w')
    yt_train = np.zeros((y_train.shape[0],2))
    yt_test = np.zeros((y_test.shape[0],2))
    yt_train[:,0] = 1 - y_train
    yt_train[:,1] = y_train
    y_train = yt_train
    yt_test[:,0] = 1 - y_test
    yt_test[:,1] = y_test
    y_test = yt_test
    trainset = TensorDataset(transform(torch.tensor(X_train,dtype=torch.float32)), torch.tensor(y_train,dtype=torch.float32))
    testset = TensorDataset(transform(torch.tensor(X_test,dtype=torch.float32)), torch.tensor(y_test,dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=16)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True,num_workers=16)
    return trainloader, testloader

def get_args_parser():
    parser = ArgumentParser(description='PyTorch Example')
    parser.add_argument('-r','--runname', type=str,default='first')
    parser.add_argument('-b','--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('-m','--model', type=str, default='resnet18')
    parser.add_argument('-d', '--device', type=str, default='cuda:14')
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.0)
    parser.add_argument('-wd','--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0, metavar='S',)
    args = parser.parse_args()
    return args

def run_one_epoch(dataloader,lossfn,model,optimizer,device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def run_one_epoch_inception(dataloader,lossfn,model,optimizer,device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs, aux_logits = model(inputs)
        loss1 = lossfn(outputs, labels)
        loss2 = lossfn(aux_logits, labels)
        loss = loss1 + 0.4*loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def test_model(dataloader, model, lossfn, device):  
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            loss = lossfn(outputs, labels)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total_loss += loss.item()
    return correct / total , total_loss


def train(args):
    
    model = modelmap[args.model]()
    device = torch.device(f'{args.device}')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainloader, testloader = load_data(args)
    model.to(device)
    Path(f'Baselines/{args.runname}').mkdir(exist_ok=True, parents=True)
    mlflow.set_experiment('Baselines')
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params(vars(args))
    
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        if args.model == 'inception':
            running_loss = run_one_epoch_inception(trainloader,criterion,model,optimizer,device)
        else:
            running_loss = run_one_epoch(trainloader,criterion,model,optimizer,device)
        acc, test_loss = test_model(testloader, model, criterion, device)
        mlflow.log_metric("train_loss", running_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", acc, step=epoch)
            
    torch.save(model.state_dict(), f'Baselines/{args.runname}/last.pth')
    mlflow.end_run()
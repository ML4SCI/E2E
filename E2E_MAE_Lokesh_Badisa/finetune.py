import torch
import torch.nn as nn
import mlflow
from pathlib import Path
import argparse
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, TensorDataset
from model import ViTMAE
import numpy as np
from einops import rearrange
from torchvision import transforms
from ericmodel import ViTMAE as EricViTMAE
from utils import *

transform = transforms.Normalize(mean=[0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 ,
       1.1099757 , 1.15771   , 1.1159292 ], std=[ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 ,
       12.441227 , 12.12112  , 11.721005 ])


optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def load_weights(wpath):
    weights = torch.load(wpath)
    for key in list(weights.keys()):
        weights[key.replace('module.','')] = weights.pop(key)
        weights[key.replace('model.','')] = weights.pop(key)
    return weights

def load_data(args):
    data = h5py.File('Dataset_Specific_labelled_train_test.h5', 'r')
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
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('-r','--runname', type=str,default='first')
    parser.add_argument('-w','--wpath', type=str,default='./maskrat15/last.pt')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('-d', '--device', type=str, default='cuda:1')
    parser.add_argument('--model', type=str, default='ericvit')
    parser.add_argument('-ls','--label_smoothing', type=float, default=0.0)
    args = parser.parse_args()
    return args

def run_one_epoch(epoch,dataloader,lossfn,model,optimizer,device,scheduler):
    model.train()
    running_loss = 0.0
    for batch_idx,data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        scheduler.step(epoch + batch_idx / len(dataloader))
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


def train(args, model):
    
    device = torch.device(f'{args.device}')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing    )
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, 5))
    trainloader, testloader = load_data(args)
    model.to(device)
    Path(f'FineTuning Experiments/{args.runname}').mkdir(exist_ok=True, parents=True)
    mlflow.set_experiment('FineTuning Experiments')
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params(vars(args))
    
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        running_loss = run_one_epoch(epoch,trainloader,criterion,model,optimizer,device,scheduler)
        acc, test_loss = test_model(testloader, model, criterion, device)
        mlflow.log_metric("train_loss", running_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", acc, step=epoch)
        
            
    torch.save(model.state_dict(), f'FineTuning Experiments/{args.runname}/model_finetuned.pth')
    mlflow.end_run()

    
class FineTunedModel(nn.Module):
    def __init__(self,model_name,weights_path=None):
        super(FineTunedModel, self).__init__()
        self.model = ViTMAE(embed_dim=768,drop_path=0.1).cpu() if model_name == 'vitmae' else EricViTMAE(drop_path=0.1).cpu()
        if weights_path is not None:
            self.model.load_state_dict(load_weights(weights_path))
        self.model.decoder_embed = nn.Identity()
        self.mask_token = nn.Identity()
        self.decoder_pos_embed = nn.Identity()
        self.decoder_blocks = nn.Identity()
        self.decoder_norm = nn.Identity()
        self.decoder_pred = nn.Identity()
        self.fc = nn.Linear(626*768, 2)
        
    def forward(self, x):
        x = self.model.forward_encoder(x,0.5,True)
        x = x.view(x.size(0),-1)
        return self.fc(x)


if __name__ == "__main__":
    args = get_args_parser()
    
    model = FineTunedModel(args.model,args.wpath)
    train(args, model)
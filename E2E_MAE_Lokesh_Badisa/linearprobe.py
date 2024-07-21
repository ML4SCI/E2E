from pyexpat import model
from finetune import load_weights,load_data,\
                    get_args_parser,test_model
from torch import nn
import torch
from pathlib import Path
import mlflow
from tqdm import tqdm
from model import ViTMAE
from torchvision import transforms
from ericmodel import ViTMAE as EricViTMAE
from utils import *

optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

transform = transforms.Normalize(mean=[0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 ,
       1.1099757 , 1.15771   , 1.1159292 ], std=[ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 ,
       12.441227 , 12.12112  , 11.721005 ])

class LinearProbeModel(nn.Module):
    def __init__(self,model_name,weights_path=None):
        super(LinearProbeModel, self).__init__()
        self.model = ViTMAE(embed_dim=256,drop_path=0.1).cpu() if model_name == 'vitmae' else EricViTMAE(drop_path=0.1).cpu()
        self.model.load_state_dict(load_weights(weights_path))
        self.model.decoder_embed = nn.Identity()
        self.mask_token = nn.Identity()
        self.decoder_pos_embed = nn.Identity()
        self.decoder_blocks = nn.Identity()
        self.decoder_norm = nn.Identity()
        self.decoder_pred = nn.Identity()
        self.fc = nn.Linear(256, 2)

    def forward(self,x):
        x = self.model.forward_encoder(x,0.5,False)
        x = x[:,1:,:].mean(dim=1)
        return self.fc(x)
    
def run_one_epoch(epoch,dataloader,lossfn,model,optimizer,device,scheduler):
    model.eval()
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

def train(args, model):
    device = torch.device(f'{args.device}')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing    )
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, 5))
    trainloader, testloader = load_data(args)
    model.to(device)

    Path(f'LinearProbing Experiments/{args.runname}').mkdir(exist_ok=True, parents=True)
    mlflow.set_experiment('LinearProbing Experiments')
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

if __name__ == '__main__':
    args = get_args_parser()

    M = LinearProbeModel(args.model,args.wpath)
    for param in M.parameters():
        param.requires_grad = False
    for param in M.fc.parameters():
        param.requires_grad = True
    train(args, M)
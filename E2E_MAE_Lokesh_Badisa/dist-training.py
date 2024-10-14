#Torch imports
import torch
from torch import nn
from einops import rearrange    
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ViTMAE
from timm.optim.optim_factory import param_groups_weight_decay

# Miscellanous imports
import time
import mlflow
from pathlib import Path   
from argparse import ArgumentParser 
import configs.model_cfg as cfg
from tqdm import tqdm, trange
from utils.optim import *
from utils.misc import *
from utils.dataset import prepare_dataloader


# Distributed training imports
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, get_world_size
import torch.distributed as dist    
import os

def get_args_parser():
    
    parser = ArgumentParser()
    parser.add_argument('--runname', type=str, required=True)
    parser.add_argument('--model', type=str, default='vitmae')
    parser.add_argument('--data_dir', type=str, default='/global/cfs/cdirs/m4392/ACAT_Backup/Data/QG/Quark_Gluon.h5')
    parser.add_argument('--base_dir', type=str, default='./')

    # GPU Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--blr', type=float, default=2e-4)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--accum_iter', type=int, default=1)

    # Training Hyperparameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mask_ratio', type=str, default=0.75)
    parser.add_argument('--norm_pix_loss', action='store_true') 
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('-cm', '--centre_masking',action='store_true')
    parser.add_argument('--rrc',action='store_true')
    parser.add_argument('--rhf',action='store_true')

    # Model Hyperparameters
    parser.add_argument('--config', type=str)
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--patch_embed', type=str, default='conv', choices=['conv', 'depthwise'])
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--decoder_embed_dim', type=int, default=128)
    parser.add_argument('--encoder_embed_dim', type=int, default=768)
    parser.add_argument('--encoder_heads', type=int, default=12)
    parser.add_argument('--decoder_heads', type=int, default=8)
    
    args = parser.parse_args()

    if args.config is not None:
        assert args.config in ['small', 'base', 'large', 'huge'], "Invalid config"
        for key in eval(f'cfg.{args.config}'):
            exec(f'args.{key} = cfg.{args.config}[\'{key}\']')
    
    # for arg in ['norm_pix_loss']:
    #     setattr(args, arg, True)

    if args.base_dir == './':
        args.base_dir = f'./Pretraining/{args.model}/{args.runname}' 

    return args

def get_transform(args):
    list_of_transforms = []
    if args.rrc:
        list_of_transforms.append(transforms.RandomResizedCrop(125, scale=(0.2, 1.0), interpolation=3))
    if args.rhf:
        list_of_transforms.append(transforms.RandomHorizontalFlip())
    
    if Path(args.data_dir).parent.stem == 'QG':
        list_of_transforms.append(transforms.Normalize(mean=QUARK_GLUON_MEAN, std=QUARK_GLUON_STD))
    else:
        list_of_transforms.append(transforms.Normalize(mean=BOOSTED_TOP_MEAN, std=BOOSTED_TOP_STD))

    return transforms.Compose(list_of_transforms)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        loss_scaler: torch.cuda.amp.GradScaler,
        optimizer: torch.optim.Optimizer,   
        args: dict
    ) -> None:
        self.local_rank = int(os.environ['SLURM_LOCALID'])
        self.global_rank = int(os.environ["SLURM_PROCID"])  
        

        self.model = model.to(f'cuda:{self.local_rank}')
        self.model = DDP(model, device_ids=[torch.device(f'cuda:{self.local_rank}')])
        
        self.optimizer = optimizer
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_scaler = loss_scaler  
        self.base_dir = args.base_dir
        self.epochs_run = 0
        self.args = args
        self.accum_iter = args.accum_iter
        self.global_min_loss = float("inf")
        # self.transform = get_transform(args)

        self.last_epoch_loss = float("inf")
        self.current_epoch_loss = 0.
        
        if self.global_rank == 0:
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_experiment('Pretraining')
            mlflow.start_run(run_name=args.runname)
            mlflow.log_params(vars(args))   


        if Path(f"{self.base_dir}/snapshot.ckpt").exists():
            print("Loading snapshot")   
            self._load_snapshot()

    def _extra_transform(self,data):
        # min = torch.min(data.view(data.size(0),-1), dim=1)[0]
        # max = torch.max(data.view(data.size(0),-1), dim=1)[0]
        # data = (data - min.view(-1, 1, 1, 1)) / (max.view(-1, 1, 1, 1) - min.view(-1, 1, 1, 1)) 
        data = rearrange(data, 'b h w c-> b c h w')
        # data = self.transform(data)
        return data        

    def _load_snapshot(self):
        snapshot = torch.load(f'{self.base_dir}/snapshot.ckpt', map_location='cpu')
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.global_min_loss = snapshot["GLOBAL_MIN_LOSS"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.loss_scaler.load_state_dict(snapshot["SCALER_STATE"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, data, update_grad):        
        with torch.cuda.amp.autocast():
            loss,_,_ = self.model(data,mask_ratio=float(self.args.mask_ratio))
        loss /= self.accum_iter
        self.loss_scaler(loss, self.optimizer, update_grad)
        return loss.item()

    def _run_epoch(self, epoch):
        loss = 0.
        dsize = 0
        self.model.train()  
        iterator = tqdm(self.train_dataloader) if self.global_rank == 0 else self.train_dataloader
        for batch, (data, _) in enumerate(iterator):
            if batch % self.accum_iter == 0:     
                self.optimizer.zero_grad()           
                adjust_learning_rate(self.optimizer, batch / len(self.train_dataloader) + epoch, self.args)
            
            
            data = data.to(torch.device(f'cuda:{self.local_rank}'))
            data = self._extra_transform(data) 
            loss += self._run_batch(data, (batch+1)%self.accum_iter==0)
            dsize += data.shape[0]
        return loss, dsize

    def _log(self, epoch, loss):
        if loss < self.global_min_loss:
            self.global_min_loss = loss
            torch.save(self.model.module.state_dict(), f"{self.base_dir}/best.pt")
        mlflow.log_metric('train_loss', loss, step=epoch)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["GLOBAL_MIN_LOSS"] = self.global_min_loss
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        snapshot["SCALER_STATE"] = self.loss_scaler.state_dict()
        torch.save(snapshot, f"{self.base_dir}/snapshot.ckpt")
        print(f"Epoch {epoch} | Training snapshot saved at {self.base_dir}/snapshot.ckpt")

    def train(self, max_epochs: int):
        if self.global_rank == 0:
            pbar = trange(self.epochs_run, max_epochs)

        for epoch in range(self.epochs_run, max_epochs):
            
            loss, dsize = self._run_epoch(epoch)

            total_loss_tensor = torch.tensor(loss).to(f'cuda:{self.local_rank}')
            total_dsize_tensor = torch.tensor(dsize).to(f'cuda:{self.local_rank}')
            dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_dsize_tensor, dst=0, op=dist.ReduceOp.SUM)

            if self.global_rank == 0:
                average_loss = total_loss_tensor.item() / total_dsize_tensor.item()
                self._save_snapshot(epoch)
                self._log(epoch, average_loss)
                pbar.update(1)

                self.current_epoch_loss = average_loss
                if self.current_epoch_loss < self.last_epoch_loss:
                    torch.save(self.model.module.state_dict(), f"{self.base_dir}/onegoodmodel.pt")
                self.last_epoch_loss = self.current_epoch_loss

            avg_loss = self.evaluate(epoch)
            # if torch.isnan(torch.Tensor(avg_loss)):
            #     break

        if self.global_rank == 0:    
            torch.save(self.model.module.state_dict(), f"{self.base_dir}/last.pt")
            Path(f"{self.base_dir}/snapshot.ckpt").unlink()
            mlflow.end_run()
    
    def evaluate(self,epoch):
        val_loss = 0.
        dsize = 0
        self.model.eval()
        with torch.no_grad():
            for _, (data, _) in enumerate(self.valid_dataloader):
                dsize += data.shape[0]
                data = data.to(torch.device(f'cuda:{self.local_rank}'))
                data = self._extra_transform(data)
                loss,_,_ = self.model(data,mask_ratio=float(self.args.mask_ratio))
                val_loss += loss.item()
            total_loss_tensor = torch.tensor(val_loss).to(f'cuda:{self.local_rank}')
            total_dsize_tensor = torch.tensor(dsize).to(f'cuda:{self.local_rank}')
            dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_dsize_tensor, dst=0, op=dist.ReduceOp.SUM)
            average_loss = total_loss_tensor.item() / total_dsize_tensor.item()
            if self.global_rank == 0:                
                mlflow.log_metric('val_loss', average_loss, step=epoch)
                print(f"Validation Loss: {average_loss}")
            return average_loss
                
        

def load_train_objs(args):
    if args.model.lower() == 'vitmae':
        model = ViTMAE(encoder_embed_dim=args.encoder_embed_dim,
                    decoder_embed_dim=args.decoder_embed_dim,
                    depth=args.encoder_depth,
                    decoder_depth=args.decoder_depth,
                    encoder_heads=args.encoder_heads,
                    decoder_heads=args.decoder_heads,
                    norm_pix_loss=args.norm_pix_loss,
                    in_chans=3 if Path(args.data_dir).parent.stem == 'QG' else 8
                    )  
        if args.patch_embed == 'depthwise':
            k_factor = args.encoder_embed_dim / model.in_chans
            list_of_layers = [
                nn.Conv2d(
                in_channels=model.in_chans, groups=model.in_chans, out_channels=model.in_chans*int(k_factor), kernel_size=args.patch_size, stride=args.patch_size
            )]
            if not k_factor.is_integer():
                list_of_layers.append(nn.Conv2d(in_channels=model.in_chans*k_factor, out_channels=args.encoder_embed_dim, kernel_size=1))
            model.patch_embed.proj = nn.Sequential(*list_of_layers)
    loss_scaler = NativeScaler()
    param_groups = param_groups_weight_decay(model, args.weight_decay)
    if args.optim.lower() != 'sgd':
        optimizer = optimizers_map[args.optim.lower()](param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = optimizers_map[args.optim.lower()](param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss_scaler, optimizer


def main(args):
    
    set_seed(args.seed)
    ddp_setup()
    eff_batch_size = args.batch_size * get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    model, loss_scaler, optimizer = load_train_objs(args)
    train_loader, val_loader, test_loader = prepare_dataloader(args.data_dir, args.batch_size)
    trainer = Trainer(model, train_loader, val_loader, loss_scaler, optimizer, args)
    trainer.train(args.epochs)
    destroy_process_group()


if __name__ == "__main__":

    args = get_args_parser()
      
    main(args)
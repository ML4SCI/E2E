#Torch imports
import torch
from torch import nn
from einops import rearrange    
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ViTMAE
from baselines.utils import modelmap
from timm.optim.optim_factory import param_groups_weight_decay

# Miscellanous imports
import numpy as np
import mlflow
from sklearn import metrics
from pathlib import Path   
from argparse import ArgumentParser 
import configs.model_cfg as cfg
from tqdm import tqdm, trange
from utils.optim import *
from utils.misc import *
from utils.dataset import H5Dataset
from torch.utils.data.distributed import DistributedSampler


# Distributed training imports
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, get_world_size
import torch.distributed as dist    
import os

def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir,exist_ok=True)
        sync_file = 'file://%s/pytorch_sync.%s' % (
                sync_file_dir, os.environ['SLURM_JOB_ID'])
        return sync_file

def ddp_setup_single_node(args):
    # Number of GPUs per node
    world_size = torch.cuda.device_count()

    # Set environment variables for PyTorch DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(args.local_rank)  # Use local rank for single-node training
    os.environ['WORLD_SIZE'] = str(world_size)  # Set total number of GPUs (world size)
    args.world_size = world_size    

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method=_get_sync_file(), world_size=world_size, rank=args.local_rank)

def prepare_dataloader(data_dir: str, batch_size: int):
    trainset = H5Dataset(data_dir, 'train')
    valset = H5Dataset(data_dir, 'validation')
    testset = H5Dataset(data_dir, 'test')
    print("Data Loaded")
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    GLOBAL_RANK = int(os.environ['RANK'])


    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(trainset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK)
        # sampler = DistributedSampler(trainset)
    )
    val_loader = DataLoader(
        valset,
        batch_size=512,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(valset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=False)
        # sampler = DistributedSampler(valset)
    )
    test_loader = DataLoader(
        testset,
        batch_size=512,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(testset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=False)
    )
    return train_loader, val_loader, test_loader

def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument('--runname', type=str, required=True)
    parser.add_argument('--model', type=str, default='vitmae')
    parser.add_argument('--data_dir', type=str, default='/global/cfs/cdirs/m4392/ACAT_Backup/Data/QG/Quark_Gluon.h5')
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DistributedDataParallel')

    # GPU Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--blr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--accum_iter', type=int, default=1)

    # Training Hyperparameters
    parser.add_argument('--lp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--centre_masking', action='store_true')
    parser.add_argument('--rrc', action='store_true')
    parser.add_argument('--rhf', action='store_true')

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

    if args.base_dir == './':
        args.base_dir = f'./Sup-Training/{args.model}/{args.runname}'

    return args

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        args: dict
    ) -> None:
        self.local_rank = args.local_rank
        self.global_rank = args.local_rank

        # Move model to GPU and wrap with DDP
        self.model = model.to(f'cuda:{self.local_rank}')
        self.model = DDP(self.model, device_ids=[args.local_rank])
        self.epochs_run = 0

        self.optimizer = optimizer
        self.args = args

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = nn.BCEWithLogitsLoss()
        self.accum_iter = args.accum_iter
        self.global_min_loss = float("inf")
        self.base_dir = args.base_dir

        if self.global_rank == 0:
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_experiment('Supervised-Training')
            mlflow.start_run(run_name=args.runname)
            mlflow.log_params(vars(args))

        if args.weights is not None:
            print("Loading weights")
            self.model.module.load_state_dict(torch.load(args.weights), strict=False)

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
        # self.loss_scaler.load_state_dict(snapshot["SCALER_STATE"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, data, labels, update_grad):     
        with torch.cuda.amp.autocast():
            print(data.device)
            pred = self.model(data)
        flabels = torch.zeros_like(pred)
        flabels[torch.arange(pred.size(0)), labels] = 1
        flabels = flabels.to(torch.device(f'cuda:{self.local_rank}'))
        loss = self.criterion(pred, flabels)
        loss /= self.accum_iter
        # self.loss_scaler(loss, self.optimizer, update_grad)
        loss.backward()
        if update_grad:
            self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        loss = 0.
        dsize = 0
        self.model.train()  
        iterator = tqdm(self.train_dataloader) if self.global_rank == 0 else self.train_dataloader
        for batch, (data, labels) in enumerate(iterator):
            if batch % self.accum_iter == 0:     
                self.optimizer.zero_grad()           
                adjust_learning_rate(self.optimizer, batch / len(self.train_dataloader) + epoch, self.args)
            
            
            data = data.to(torch.device(f'cuda:{self.local_rank}'))
            labels = labels.long().to(torch.device(f'cuda:{self.local_rank}'))
            data = self._extra_transform(data) 
            loss += self._run_batch(data, labels, (batch+1)%self.accum_iter==0)
            dsize += data.shape[0]
        return loss, dsize

    def _log(self, epoch, loss):
        if loss < self.global_min_loss:
            self.global_min_loss = loss
            torch.save(self.model.module.state_dict(), f"{self.base_dir}/best_trainloss.pt")
        mlflow.log_metric('train_loss', loss, step=epoch)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["GLOBAL_MIN_LOSS"] = self.global_min_loss
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        # snapshot["SCALER_STATE"] = self.loss_scaler.state_dict()
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

            self.evaluate(epoch,'val')
        
        self.evaluate(max_epochs,'test')

        if self.global_rank == 0:    
            torch.save(self.model.module.state_dict(), f"{self.base_dir}/last.pt")
            Path(f"{self.base_dir}/snapshot.ckpt").unlink()
            mlflow.end_run()
    
    def evaluate(self,epoch,type='val'):
        curr_loss = 0.
        dsize = 0
        data_loader = self.valid_dataloader if type == 'val' else self.test_dataloader
        self.model.eval()
        fpreds, flabels = [], []
        with torch.no_grad():
            for _, (data, labels) in enumerate(data_loader):
                dsize += data.shape[0]
                data = data.to(torch.device(f'cuda:{self.local_rank}'))
                labels = labels.long().to(torch.device(f'cuda:{self.local_rank}'))
                data = self._extra_transform(data)
                pred = self.model(data)
                new_labels = torch.zeros_like(pred)
                new_labels[torch.arange(pred.size(0)), labels] = 1
                loss = self.criterion(pred, new_labels)
                curr_loss += loss.item()
                fpreds.append(pred.softmax(dim=1))
                flabels.append(labels)

            fpreds = torch.cat(fpreds, dim=0)
            flabels = torch.cat(flabels, dim=0)
            
            # Prepare tensors for all_gather
            gathered_fpreds = [torch.zeros_like(fpreds) for _ in range(dist.get_world_size())]
            gathered_flabels = [torch.zeros_like(flabels) for _ in range(dist.get_world_size())]
            
            # Concatenate across GPUs
            dist.all_gather(gathered_fpreds, fpreds)
            dist.all_gather(gathered_flabels, flabels)
            
            # Flatten the gathered lists
            fpreds = torch.cat(gathered_fpreds, dim=0)
            flabels = torch.cat(gathered_flabels, dim=0)

            
            # Aggregate Loss
            total_loss_tensor = torch.tensor(curr_loss).to(f'cuda:{self.local_rank}')
            total_dsize_tensor = torch.tensor(dsize).to(f'cuda:{self.local_rank}')
            dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_dsize_tensor, dst=0, op=dist.ReduceOp.SUM)

            if self.global_rank == 0:
                average_loss = total_loss_tensor.item() / total_dsize_tensor.item()
                mlflow.log_metric(f'{type}_loss', average_loss, step=epoch)
                print(f"{type} Loss: {average_loss}")

                #Calculate accuracy
                acc = torch.sum(torch.argmax(fpreds, dim=1) == flabels).item() / flabels.shape[0]
                mlflow.log_metric(f'{type}_accuracy', acc, step=epoch)
                print(f"{type} Accuracy: {acc}")

                #Calculate AUC
                flabels = flabels.cpu().numpy()
                fpreds = fpreds.max(1)[0].cpu().numpy()                
                try:
                    fpr, tpr, _ = metrics.roc_curve(flabels, fpreds, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    mlflow.log_metric(f'{type}_auc', auc, step=epoch)
                    print(f"{type} AUC: {auc}")
                except:
                    print(f"Number of NaNs in predictions: {torch.isnan(fpreds).sum()}")
    
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
    else:
        model = modelmap[args.model](3 if Path(args.data_dir).parent.stem == 'QG' else 8)
    
    # loss_scaler = NativeScaler()
    param_groups = param_groups_weight_decay(model, args.weight_decay)
    if args.optim.lower() != 'sgd':
        optimizer = optimizers_map[args.optim.lower()](param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = optimizers_map[args.optim.lower()](param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # return model, loss_scaler, optimizer
    return model, optimizer


def main(args):
    set_seed(args.seed)
    ddp_setup_single_node(args)  # Set up DDP for single-node multi-GPU

    eff_batch_size = args.batch_size * args.world_size
    args.lr = args.blr * eff_batch_size / 256

    model, optimizer = load_train_objs(args)
    train_loader, val_loader, test_loader = prepare_dataloader(args.data_dir, args.batch_size)

    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, args)
    trainer.train(args.epochs)
    destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser()
    main(args)

# Torch imports
import torch
from torch import nn
from utils.optim import *
from model import VisionTransformer
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from timm.layers.weight_init import trunc_normal_

# Miscellanous imports
import mlflow
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from utils.misc import *
from utils.dataset import prepare_dataloader
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.optim import adjust_learning_rate, optimizers_map


# Distributed training imports
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, get_world_size
import torch.distributed as dist
import os


def get_args_parser():
    parser = ArgumentParser(description='PyTorch Example')

    # Model hyperparameters
    parser.add_argument('-r','--runname', type=str,default='first')
    parser.add_argument('-w','--weights', type=str,default='./Pretraining/vitmae/vit_base/best.pt')
    parser.add_argument('--model', type=str, default='vitmae')
    parser.add_argument('--config', type=str)
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--patch_embed', type=str, default='conv', choices=['conv', 'depthwise'])
    parser.add_argument('--encoder_embed_dim', type=int, default=768)
    parser.add_argument('--encoder_heads', type=int, default=12)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--patch_size', type=int, default=5)

    # Data Hyperparameters
    parser.add_argument('--data_dir', type=str, default='/global/cfs/cdirs/m4392/ACAT_Backup/Data/QG/Quark_Gluon.h5')
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=1024)

    # GPU Hyperparameters
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--blr', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    # Optimizer Hyperparameters 
    parser.add_argument('--optim', type=str, default='lars', choices=['lars', 'adam', 'sgd', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    args = parser.parse_args()
    args.global_pool = True

    if args.config is not None:
        assert args.config in ['small', 'base', 'large', 'huge'], "Invalid config"
        for key in eval(f'cfg.{args.config}'):
            exec(f'args.{key} = cfg.{args.config}[\'{key}\']')
    else:
        assert args.encoder_embed_dim is not None, "Provide encoder_embed_dim"
        assert args.encoder_heads is not None, "Provide encoder_heads"
        assert args.encoder_depth is not None, "Provide encoder_depth"

    if args.base_dir == './':
        args.base_dir = f"{args.model}/{args.runname}"
    return args


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        loss_scaler: torch.cuda.amp.GradScaler,
        args: dict
    ) -> None:
        self.local_rank = int(os.environ['SLURM_LOCALID'])
        self.global_rank = int(os.environ["SLURM_PROCID"])  
        self.model = model.to(self.local_rank)  # equivalent to `output_device` in DDP
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = CrossEntropyLoss()
        self.loss_scaler = loss_scaler  
        self.base_dir = args.base_dir
        self.epochs_run = 0
        self.args = args
        self.train_transform = get_transform(args, 'train')
        self.valid_transform = get_transform(args, 'valid')
        self.global_min_loss = float("inf")
        if Path(f"Linear-Probing/{self.base_dir}/snapshot.ckpt").exists():
            print("Loading snapshot")   
            self._load_snapshot(f"Linear-Probing/{self.base_dir}/snapshot.ckpt")
        
        if self.local_rank == 0:
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_experiment('Linear Probing')
            mlflow.start_run(run_name=args.runname)
            mlflow.log_params(vars(args))   

        self.model = DDP(self.model, device_ids=[self.local_rank])
        model_without_ddp = self.model.module   
        if args.optim.lower() != 'sgd':
            self.optimizer = optimizers_map[args.optim.lower()](model_without_ddp.head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        elif args.optim.lower() != 'lars':
            self.optimizer = optimizers_map[args.optim.lower()](model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optimizers_map[args.optim.lower()](model_without_ddp.head.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def _load_snapshot(self):
        snapshot = torch.load(f'Linear-Probing/{self.base_dir}/snapshot.ckpt', map_location='cpu')
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.global_min_loss = snapshot["GLOBAL_MIN_LOSS"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.loss_scaler.load_state_dict(snapshot["SCALER_STATE"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, data, labels):
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = self.model(data,mask_ratio=float(self.args.mask_ratio))
            loss = self.criterion(pred, labels)
        self.loss_scaler(loss, self.optimizer)
        return loss.item()

    def _run_epoch(self, epoch):
        loss = 0
        dsize = 0
        self.model.train()  
        for batch, (data, labels) in enumerate(self.train_dataloader):
            adjust_learning_rate(self.optimizer, batch / len(self.train_dataloader) + epoch, self.args)
            data = data.to(self.local_rank)
            data = self.train_transform(data)
            loss += self._run_batch(data, labels)
            dsize += data.shape[0]
        return loss, dsize

    @torch.no_grad
    def _evaluate(self, dataloader):
        self.model.eval()
        y_true_local, y_pred_local = [], []
        
        for data, target in dataloader:
            data, target = data.to(self.local_rank), target.to(self.local_rank)
            data = self.valid_transform(data)
            output = self.model(data)
            y_true_local.extend(target.cpu().numpy())
            y_pred_local.extend(output.max(dim=1)[1].cpu().numpy())
        
        # Gather predictions and targets from all processes
        y_true_gathered = [None for _ in range(dist.get_world_size())]
        y_pred_gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(y_true_gathered, y_true_local)
        dist.all_gather_object(y_pred_gathered, y_pred_local)

        # Concatenate results from all processes
        y_true = [item for sublist in y_true_gathered for item in sublist]
        y_pred = [item for sublist in y_pred_gathered for item in sublist]

        # Calculate accuracy and ROC-AUC score on the root process
        if self.local_rank == 0:
            acc = accuracy_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            return acc, roc_auc
        return None, None

    def _log(self, epoch, loss):
        if loss < self.global_min_loss:
            self.global_min_loss = loss
            torch.save(self.model.module.state_dict(), f"Linear-Probing/{self.base_dir}/best.pt")
        
        train_acc, train_roc_auc = self._evaluate(self.train_dataloader)
        valid_acc, valid_roc_auc = self._evaluate(self.valid_dataloader)

        # Only log metrics if we're on the root process
        if self.local_rank == 0:
            mlflow.log_metric('loss', loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_acc, step=epoch)
            mlflow.log_metric('train_roc_auc', train_roc_auc, step=epoch)
            mlflow.log_metric('valid_accuracy', valid_acc, step=epoch)
            mlflow.log_metric('valid_roc_auc', valid_roc_auc, step=epoch)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["GLOBAL_MIN_LOSS"] = self.global_min_loss
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        snapshot["SCALER_STATE"] = self.loss_scaler.state_dict()
        torch.save(snapshot, f"Linear-Probing/{self.base_dir}/snapshot.ckpt")
        print(f"Epoch {epoch} | Training snapshot saved at {self.base_dir}/snapshot.ckpt")

    def train(self, max_epochs: int):
        with tqdm(range(self.epochs_run, max_epochs), desc="Epochs") as pbar:
            for epoch in range(self.epochs_run, max_epochs):
                self.model.train()
                loss, dsize = self._run_epoch(epoch)

                total_loss_tensor = torch.tensor(loss).to(self.local_rank)
                total_dsize_tensor = torch.tensor(dsize).to(self.local_rank)
                dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_dsize_tensor, dst=0, op=dist.ReduceOp.SUM)

                if self.local_rank == 0:
                    average_loss = total_loss_tensor.item() / total_dsize_tensor.item()
                    self._save_snapshot(epoch)
                    self._log(epoch, average_loss)
                    pbar.update(1)
            if self.local_rank == 0:    
                torch.save(self.model.module.state_dict(), f"{self.base_dir}/last.pt")
                Path(f"Linear-Probing/{self.base_dir}/snapshot.ckpt").unlink()
                mlflow.end_run()

def load_train_objs(args):
    if args.model.lower() == 'vitmae':
       
        model = VisionTransformer(
                    img_size=125,
                    patch_size=args.patch_size,   
                    embed_dim=args.encoder_embed_dim,
                    depth=args.encoder_depth,
                    num_heads=args.encoder_heads,
                    in_chans=3 if Path(args.data_dir).parent.stem == 'QG' else 8,
                    global_pool=args.global_pool,
                    num_classes=2   
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

        checkpoint = torch.load(args.weights, map_location='cpu')   
        model.load_state_dict(checkpoint, strict=False)
        trunc_normal_(model.head.weight, std=0.01)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    loss_scaler = NativeScaler()

    return model, loss_scaler


def get_transform(args, mode):
    list_of_transforms = []
    if mode == 'train':
        list_of_transforms.append(transforms.RandomResizedCrop(125, interpolation=3))
        list_of_transforms.append(transforms.RandomHorizontalFlip())
    
    if Path(args.data_dir).parent.stem == 'QG':
        list_of_transforms.append(transforms.Normalize(mean=QUARK_GLUON_MEAN, std=QUARK_GLUON_STD))
    else:
        list_of_transforms.append(transforms.Normalize(mean=BOOSTED_TOP_MEAN, std=BOOSTED_TOP_STD))



def main():
    args = get_args_parser()
    set_seed(args.seed)
    ddp_setup()
    eff_batch_size = args.batch_size * get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    model, optimizer, loss_scaler = load_train_objs(args)
    train_loader, val_loader = prepare_dataloader(args.data_dir, args.batch_size)
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_scaler, args)
    trainer.train(args.epochs)
    destroy_process_group()
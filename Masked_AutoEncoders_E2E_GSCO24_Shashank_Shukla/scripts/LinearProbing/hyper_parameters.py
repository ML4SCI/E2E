
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import argparse
from util import *
from data import *
from model import get_model
#---------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------Learning rate scheduler---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_batches_per_epoch, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch+1 < self.warmup_epochs * self.num_batches_per_epoch:
            return [base_lr * (self.last_epoch+1) / (self.warmup_epochs * self.num_batches_per_epoch) for base_lr in self.base_lrs]
        else:
            lr = [base_lr * (1 - (self.last_epoch+1 - self.warmup_epochs * self.num_batches_per_epoch) / \
                ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch)) for base_lr in self.base_lrs]
            return lr
            

#---------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------Setup of Distributed Data parallel------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
#---------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------Get Linear Model wrapped up-----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
class Get_linear_model(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes, unfreeze=False, mean_or_max='max'):
        super().__init__()
        self.mean_or_max = mean_or_max
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
             param.requires_grad = False
                
        if unfreeze:
            self.enable_layers(pretrained_model)
            
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.batchnorm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def enable_layers(self, model):
        if hasattr(self.pretrained_model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.pretrained_model.parameters():
                param.requires_grad = True

    def forward(self, x):
        if hasattr(self.pretrained_model, '_process_input'):
            x = self.pretrained_model._process_input(x)
        
        n = x.shape[0]

        if hasattr(self.pretrained_model, 'encoder'):
            enc_output, mask, ids_restore = self.pretrained_model.encoder(x, 0.)
        else:
            enc_output, mask, ids_restore = self.pretrained_model.forward_encoder(x, 0.)
           
        if isinstance(enc_output, list):
            enc_output = torch.cat(enc_output, axis = 1)
                
        if self.mean_or_max == 'max':
            x, _ = torch.max(enc_output, dim=1)
        elif self.mean_or_max == 'mean':
            x = torch.mean(enc_output, dim=1)
        x = self.batchnorm(x)

        return self.linear(x)

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------Loss Function------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

def Custom_loss(device, predicted, inputs):
    """
    imgs: [N, 8, H, W]
    pred: [N, L, p*p*8]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    accuracy = Accuracy(task = 'binary').to(device)
    auroc = AUROC(task='binary').to(device)
    criterion = nn.BCEWithLogitsLoss(reduction = "mean").to(device)
    
    loss = criterion(predicted, inputs)
    train_acc = accuracy(predicted, inputs)
    train_auroc = auroc(predicted, inputs)
    
    return loss, train_acc, train_auroc
#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Trainer Class------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#        
from tqdm import tqdm
import json
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        save_every: int,
        args,
        base_path: str = '/global/homes/s/ssshukla/scripts/',
    ) -> None:
        
        #--------------------------------------------------------------#
        self.args = args
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.models = model
        self.train_dataloader = train_data
        self.valid_dataloader = valid_data
        self.num_batches_per_epoch = len(self.train_dataloader.dataset) // torch.distributed.get_world_size() // args.batch_size
        
        #--------------------------------------------------------------#
        #Save and read checkpoints or logd from specified path
        self.save_every = 1 if save_every is None else save_every
        self.ckp_PATH = f"/global/homes/s/ssshukla/Shashank/scripts/Results/Linear_probing/Weights/{self.args.model_name}/checkpoint.pt"
        self.logs_PATH = f"/global/homes/s/ssshukla/Shashank/scripts/Results/Linear_probing/Logs/{self.args.model_name}/"
        self.weights_path = os.path.join(args.MAE_path,f"{self.args.model_name}/checkpoint.pt")
        #--------------------------------------------------------------#
          
        #--------------------------------------------------------------#
        #Load Checkpoints and wrap-up with linear linear layer on top classification purposes
        self._load_autoencoder_weight(self.weights_path)
        self.linear_model = Get_linear_model(self.models, 
                                             hidden_dim = args.embed_dim, 
                                             num_classes = args.num_classes, 
                                             unfreeze = False, 
                                             mean_or_max = args.mode).to(self.gpu_id)
        #--------------------------------------------------------------#
        
        #--------------------------------------------------------------#
        #Wrap the mode in Distributed Data Parallel
        if torch.distributed.get_world_size() > 1:
            self.model = DDP(self.linear_model, device_ids=[self.gpu_id])
        else:
            self.model = self.linear_model

        # if self.args.resume_training is True: self._load_checkpoints(self.ckp_PATH)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Unfreeze condition -- ",self.args.unfreeze)
        print("Number of Trainable params -- ",trainable_params)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate * torch.distributed.get_world_size())
        
        self.base_path = base_path
        self.scheduler = WarmupScheduler(self.optimizer, 
                                 warmup_epochs=self.args.warmup, 
                                 total_epochs=self.args.epochs,
                                 num_batches_per_epoch=self.num_batches_per_epoch)
        #--------------------------------------------------------------#
        
        #--------------------------------------------------------------#
        #Record the performance metrics for the particular model
        self.loss = []
        self.accuracy = []
        self.auc_roc = []
        self.val_loss = []
        self.val_accuracy = []
        self.val_auc_roc = []
        self.performance_characteristics = {"Train_Loss" : [], "Validation_loss" : [], "Train_accuracy" : [], "Validation_accuracy" : [], "Train_auc_roc_score" : [], "Validation_auc_roc" : []}
        #--------------------------------------------------------------#
        
    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        
    def _run_batch(self, source, labels):
        """
        Training on single batch
        Args:
            source: Input passed to the model
        """
        #--------------------------------------------------------------#
        self.optimizer.zero_grad()
        outputs = self.model(source)
        loss, acc, auc_roc = Custom_loss(self.gpu_id, outputs.squeeze(), labels)
        loss.backward()
        self.optimizer.step()
        return (loss.detach().cpu().numpy()), (acc.detach().cpu().numpy()), (auc_roc.detach().cpu().numpy())
        #--------------------------------------------------------------#

    def _run_val_batch(self, source, labels):
        """
        Validation on single batch
        Args:
            source: Input passed to the model
            labels: Ground - Truth 
        """
        #--------------------------------------------------------------#
        outputs = self.model(source)
        loss, acc, auc_roc = Custom_loss(self.gpu_id, outputs.squeeze(), labels)
        return (loss.detach().cpu().numpy()), (acc.detach().cpu().numpy()), (auc_roc.detach().cpu().numpy())
        #--------------------------------------------------------------#
        
    def _run_epoch(self, epoch):
        """
        Training on single epoch
        """
        
        #-----------------------------------------Train Epoch-------------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        print("Training Loop Started")
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_dataloader)}")

        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Train-Epoch {epoch} [GPU {self.gpu_id}]",
            ncols=100,
            leave=False
        )
        self.model.train()
        for step, batch in progress_bar:
            image = batch['img'].to(self.gpu_id)
            labels = batch['labels'].to(self.gpu_id)
            loss, accuracy, auc_roc = self._run_batch(image, labels)
            
            # Step the scheduler if applicable
            if step < self.scheduler.num_batches_per_epoch:
                self.scheduler.step()
            
            # Update progress bar description with the latest loss
            progress_bar.set_postfix(loss=loss)

            if step % 200 == 0:
                with open(f'linear_probe_logs.txt', 'a') as f:
                    f.write(f'Train -- Step {step} loss --  {loss}|| accuracy -- {accuracy} || auc_roc_score -- {auc_roc}\n')
            self.loss.append(loss)
            self.accuracy.append(accuracy)
            self.auc_roc.append(auc_roc)
            
        #---------------------------------------Validation Epoch----------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        print("Validation Loop Started")
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.valid_dataloader)}")
        # Wrap the dataloader with tqdm for a progress bar
        valid_progress_bar = tqdm(
            enumerate(self.valid_dataloader),
            total=len(self.valid_dataloader),
            desc=f"Valid-Epoch {epoch} [GPU {self.gpu_id}]",
            ncols=100,
            leave=False
        )
        self.model.eval()
        with torch.no_grad():
            for step, batch in valid_progress_bar:
                image = batch['img'].to(self.gpu_id)
                labels = batch['labels'].to(self.gpu_id)
                loss, accuracy, auc_roc = self._run_val_batch(image, labels)

                # Update progress bar description with the latest loss
                progress_bar.set_postfix(loss=loss)

                self.val_loss.append(loss)
                self.val_accuracy.append(accuracy)
                self.val_auc_roc.append(auc_roc)
    
    def _load_checkpoints(self, weights_path):
        try:
            print("Checkpoint loaded ....")
            self.model.load_state_dict(torch.load(weights_path))
        except FileNotFoundError:
            print("Checkpoint file not found.")
            
    def _load_autoencoder_weight(self, weights_path):
        try:
            print("Weights loaded ....")
            self.models.load_state_dict(torch.load(weights_path))
        except FileNotFoundError:
            print("Weights file not found.")
        
    def _save_checkpoint(self, epoch, trn_loss, trn_acc, trn_auc, val_loss, val_acc, val_auc):
        ckp = self.model.module.state_dict()
        
        #--------------------------------------------------------------#
        # Save the checkpoint at specified path
        torch.save(ckp, self.ckp_PATH)
        #--------------------------------------------------------------#
        
        #--------------------------------------------------------------#
        # Save the logs at respective path
        with open(f'{self.logs_PATH}losses.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}, Train acc: {trn_acc:.4f}, Val acc: {val_acc:.4f}, Train auc: {trn_auc:.4f}, Val auc: {val_auc:.4f}\n')
        
        self.performance_characteristics["Train_Loss"].append(trn_loss.astype('float64'))
        self.performance_characteristics["Validation_loss"].append(val_loss.astype('float64'))
        self.performance_characteristics["Train_accuracy"].append(trn_acc.astype('float64'))
        self.performance_characteristics["Validation_accuracy"].append(val_acc.astype('float64'))
        self.performance_characteristics["Train_auc_roc_score"].append(trn_auc.astype('float64'))
        self.performance_characteristics["Validation_auc_roc"].append(val_auc.astype('float64'))
        
        print(f"Epoch {epoch} | Training checkpoint saved at {self.ckp_PATH}")
        #--------------------------------------------------------------#

    def train(self, max_epochs: int):
        max_auc = 0
        for epoch in (range(max_epochs)):
            self._run_epoch(epoch)
            max_auc = max(max_auc, np.mean(self.val_auc_roc))
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, 
                                      np.mean(self.loss), 
                                      np.mean(self.accuracy), 
                                      np.mean(self.auc_roc), 
                                      np.mean(self.val_loss), 
                                      np.mean(self.val_accuracy), 
                                      np.mean(self.val_auc_roc))
                self.loss = []
                self.accuracy = []
                self.auc_roc = []
                self.val_loss = []
                self.val_accuracy = []
                self.val_auc_roc = []
                
        with open(f'{self.logs_PATH}performance_metrics.json', "w") as outfile: 
            json.dump(self.performance_characteristics, outfile)
            
        return max_auc
#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Initialisation-----------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
def load_train_objs(args):
    
    # load your dataset
    train_set = H5MaskedAutoEncoderDataset(h5_path = args.data_path, mode = 'train', preload_size = args.batch_size, n_samples = args.train_samples)
    
    #Validation Dataset
    valid_set = H5MaskedAutoEncoderDataset(h5_path = args.data_path, mode = 'validation', preload_size = args.batch_size)
    
    
    # load your model
    model = get_model(args)
    
    #Convert BatchNorm to SyncBatchNorm to sync the running stats of BatchNorm layers across replicas.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    return train_set, valid_set, model

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Dataloader Preparation---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#    
def prepare_dataloader(train_dataset: Dataset, valid_dataset: Dataset, batch_size: int, n_train: int):
    
    CHUNK_SIZE = 32
    # Helper function to chunk the indices
    def chunk_indices(indices, chunk_size):
        for i in range(0, len(indices), chunk_size):
            yield indices[i:i + chunk_size]
            
    n_total_train = len(train_dataset)
    
    if n_train != -1:
        train_indices = list(range(2000000, 2000000 + n_train))
        random.shuffle(train_indices) 
        chunked_train_indices = list(chunk_indices(train_indices, CHUNK_SIZE))
        random.shuffle(chunked_train_indices)
        train_indices = [index for chunk in chunked_train_indices for index in chunk]

    else:
        train_indices = list(range(n_total_train))
        random.shuffle(train_indices)
        chunked_train_indices = list(chunk_indices(train_indices, CHUNK_SIZE))
        random.shuffle(chunked_train_indices)
        train_indices = [index for chunk in chunked_train_indices for index in chunk]
        
    print("Length of validation set data ",len(valid_dataset))
    print("Length of train set data ",len(train_indices))
    
    #Valid Indices Calculation
    valid_indices = list(range(len(valid_dataset)))
    random.shuffle(valid_indices)
    chunked_valid_indices = list(chunk_indices(valid_indices, CHUNK_SIZE))
    random.shuffle(chunked_valid_indices)
    valid_indices = [index for chunk in chunked_valid_indices for index in chunk]
    
    # print(train_indices)
    train_sampler = ChunkedDistributedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=batch_size,
                                    num_workers = 8,
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=train_sampler,
                                    drop_last=True
                                )
    
    valid_sampler = ChunkedDistributedSampler(valid_indices, chunk_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
                                    valid_dataset,
                                    batch_size=batch_size,
                                    num_workers = 8,
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=valid_sampler,
                                    drop_last=True
                                )
    
    return train_dataloader, valid_dataloader

# DDP setup and cleanup functions remain the same
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
def destroy_process_group():
    dist.destroy_process_group()

# The model training function needs to be modified to accept hyperparameters
def train_model_with_hyperparams(world_size, hyperparams):
    global args
    ddp_setup()
    print(hyperparams)
    # Apply hyperparameters
    args.learning_rate = hyperparams['learning_rate']
    args.weight_decay = hyperparams['weight_decay']
    
    # Load dataset, model, and corresponding optimizer
    train_dataset, valid_dataset, model = load_train_objs(args)

    # Prepare data loaders
    train_dataloader, valid_dataloader = prepare_dataloader(train_dataset, valid_dataset, args.batch_size, args.train_samples)

    # Initialize Trainer and train the model
    trainer = Trainer(model, train_dataloader, valid_dataloader, args.save_every, args)
    val_auc_score = trainer.train(args.epochs)  # Assume this returns validation loss

    destroy_process_group()
    
    return val_auc_score

# Define the search space
search_space = [
    Real(1e-5, 1e-2, name='learning_rate'),
    Real(1e-6, 1e-2, name='weight_decay'),
]

# Objective function for Bayesian optimization
@use_named_args(search_space)
def objective(**params):
    world_size = torch.cuda.device_count()
    print(params)
    # Train model with the given hyperparameters and return validation loss
    val_auc_score = train_model_with_hyperparams(world_size, params)

    # Return the validation loss for skopt to minimize (e.g., -AUC score for maximization)
    return 1-val_auc_score  # Minimize the negative AUC score (or maximize AUC)

# Main function to perform Bayesian optimization using skopt
def bayesian_optimization():
    global args
    # Run Bayesian optimization
    result = gp_minimize(objective, search_space, n_calls=10, random_state=42)

    # Print best hyperparameters
    print(f"Best hyperparameters: {result.x}")
    print(f"Best validation accuracy: {1 - result.fun}")
    
    with open(f'{args.model_name}params.txt', 'a') as f:
        f.write(f'Best Hyperparameters : {result.x},  Val auc: {1 - result.fun}\n')


def get_args_parser():
    parser = argparse.ArgumentParser('Masked Autoencoder ViT', add_help=False)

    # Model related arguments
    parser.add_argument('--model_name', default="base_mae_depthwise_convolution", choices=["base_mae_depthwise_convolution",
                                                                                           "channel_former",
                                                                                           "base_mae",
                                                                                           "conv_mae",
                                                                                           "cross_vit"],type=str, help='Model architecture to train')
    parser.add_argument('--img_size', default=125, type=int, help='Image size')
    parser.add_argument('--patch_size', default=5, type=int, help='Patch size')
    parser.add_argument('--in_chans', default=8, type=int, help='Number of input channels')
    parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dimension')
    parser.add_argument('--depth', default=16, type=int, help='Depth of the encoder')
    parser.add_argument('--num_heads', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--k_factor', default=16, type=int, help='Factor for convolution projection')

    # Decoder related arguments
    parser.add_argument('--decoder_embed_dim', default=128, type=int, help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=8, type=int, help='Decoder depth')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='Number of decoder heads')

    # Other arguments
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio')
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=str, help='Normalization layer')
    parser.add_argument('--mlp_ratio', default=4, type=float, help='MLP ratio')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--train_samples', default=-1, type=int, help='-1 indicates use samples for training')
    parser.add_argument('--resume_training', default=False, type=bool, help='Weather to resume from a checkpoint')
    parser.add_argument('--data_path', default='/pscratch/sd/s/ssshukla/Boosted_Top.h5', type=str, help='Path to the dataset')
    parser.add_argument('--MAE_path', default='/global/homes/s/ssshukla/Shashank/scripts/Results/Weights/', type=str, help='Path to the autoencoder')
    parser.add_argument('--warmup', type=int, default=3, ###This should be ~5-10% of total epochs
                        help='number of warmup epochs before reaching base_lr')
    parser.add_argument('--num_classes', default=1, type=int, help='Type of classification problem to be solved')
    parser.add_argument('--unfreeze', default=False, type=bool, help='Linear Probing or finetuning')
    parser.add_argument('--mode', default="mean", type=str, help='Mean pooling or max pooling before linear layer')
    parser.add_argument('--weight_decay', default=0.00005, type=float, help='weight_decay')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE ViT Training and Testing', parents=[get_args_parser()])
    global args
    args = parser.parse_args()
    
    # Perform Bayesian optimization
    bayesian_optimization()

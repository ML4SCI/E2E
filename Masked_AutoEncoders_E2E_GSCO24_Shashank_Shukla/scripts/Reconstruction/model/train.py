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
#--------------------------------------------------------Loss Function------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

def Custom_loss(rank, predicted, inputs):
        """
        imgs: [N, 8, H, W]
        pred: [N, L, p*p*8]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        def patchify(imgs):
            """
            imgs: (N, 8, H, W)
            x: (N, L, patch_size**2 *8)
            """
            p = 5
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 8, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 8))
            return x
        
        target = patchify(inputs)
        criterion = nn.BCEWithLogitsLoss(reduction = "mean").to(rank)
        loss = criterion(predicted, target)
        return loss
#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Trainer Class------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#        
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        # gpu_id: int,
        save_every: int,
        args,
        base_path: str = '/global/homes/s/ssshukla/scripts/',
    ) -> None:
        self.args = args
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_dataloader = train_data
        self.num_batches_per_epoch = len(self.train_dataloader.dataset) // torch.distributed.get_world_size() // args.batch_size
        
        self.save_every = 1 if save_every is None else save_every
        self.ckp_PATH = f"/global/homes/s/ssshukla/Shashank/scripts/Results/Weights/{self.args.model_name}/checkpoint.pt"
        self.logs_PATH = f"/global/homes/s/ssshukla/Shashank/scripts/Results/Logs/{self.args.model_name}/"
        
        if args.resume_training is True: self._load_checkpoints(self.ckp_PATH)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * torch.distributed.get_world_size())
        self.model.train()
        self.base_path = base_path
        self.scheduler = WarmupScheduler(self.optimizer, 
                                 warmup_epochs=args.warmup, 
                                 total_epochs=args.epochs,
                                 num_batches_per_epoch=self.num_batches_per_epoch)
        self.loss = []
        
        
    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        
    def _run_batch(self, source):
        """
        Training on single batch
        Args:
            source: Input passed to the model
        """
        self.optimizer.zero_grad()
        loss, outputs, mask = self.model(source)
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.detach().cpu().numpy())

    def _run_epoch(self, epoch):
        """
        Training on single epoch
        """
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_dataloader)}")

        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch} [GPU {self.gpu_id}]",
            ncols=100,
            leave=False
        )

        for step, batch in progress_bar:
            image = batch['img'].to(self.gpu_id)
            self._run_batch(image)
            
            # Step the scheduler if applicable
            if step < self.scheduler.num_batches_per_epoch:
                self.scheduler.step()
            
            # Update progress bar description with the latest loss
            progress_bar.set_postfix(loss=self.loss[-1])

            if step % 200 == 0:
                with open(f'logs.txt', 'a') as f:
                    f.write(f'Step {step} loss --  {self.loss[-1]}\n')
    
    def _load_checkpoints(self, weights_path):
        try:
            print("Checkpoint loaded ....")
            self.model.load_state_dict(torch.load(weights_path))
        except FileNotFoundError:
            print("Checkpoint file not found.")
        
    def _save_checkpoint(self, epoch, trn_loss):
        ckp = self.model.module.state_dict()
        
        # Save the checkpoint at specified path
        torch.save(ckp, self.ckp_PATH)
        
        # Save the logs at respective path
        with open(f'{self.logs_PATH}losses.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}\n')
            
        print(f"Epoch {epoch} | Training checkpoint saved at {self.ckp_PATH}")

    def train(self, max_epochs: int):
        for epoch in (range(max_epochs)):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, np.mean(self.loss))
                self.loss = []

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Initialisation-----------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
def load_train_objs(args):
    
    # load your dataset
    train_set = H5MaskedAutoEncoderDataset(h5_path = args.data_path, preload_size = args.batch_size)
    
    # load your model
    model = get_model(args)
    
    #Convert BatchNorm to SyncBatchNorm to sync the running stats of BatchNorm layers across replicas.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    return train_set, model

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Dataloader Preparation---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#    
def prepare_dataloader(dataset: Dataset, batch_size: int, n_train: int):
    
    CHUNK_SIZE = 32
    # Helper function to chunk the indices
    def chunk_indices(indices, chunk_size):
        for i in range(0, len(indices), chunk_size):
            yield indices[i:i + chunk_size]
            
    n_total_train = len(dataset)
    
    if n_train != -1:
        train_indices = list(range(n_train))
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
        
    # print(train_indices)
    train_sampler = ChunkedDistributedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(
                                    dataset,
                                    batch_size=batch_size,
                                    num_workers = 8,
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=train_sampler,
                                    drop_last=True
                                )
    return train_dataloader

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Spwaning function--------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
def main(world_size: int, args):
    
    # Setup the Distributed Data Parallel
    ddp_setup()
    
    # Load dataset, model and corresponding optimizer
    dataset, model = load_train_objs(args)
    
    # Prepare the dataloader
    train_dataloader = prepare_dataloader(dataset, args.batch_size, args.train_samples)
    
    # Initialize the Trainer Class
    trainer = Trainer(model, train_dataloader,  args.save_every, args)
    
    #Start Training
    trainer.train(args.epochs)
    
    #Kill the process
    destroy_process_group()

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Arguments----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
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
    parser.add_argument('--warmup', type=int, default=3, ###This should be ~5-10% of total epochs
                        help='number of warmup epochs before reaching base_lr')
    return parser

#---------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Start Training-----------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE ViT Training and Testing', parents=[get_args_parser()])
    args = parser.parse_args()

    with open(f'logs.txt', 'a') as f:
        f.write(f'Torch version --  {torch.__version__}\n')
            
    world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args), nprocs=world_size)
    main(world_size, args)
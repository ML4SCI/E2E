#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -n 1                # Number of tasks
#SBATCH -G 2                # Request 2 GPUs
#SBATCH -C gpu&hbm80g       # Ensure we get GPU nodes
#SBATCH --cpus-per-task=64
#SBATCH -q debug            # Queue (e.g., regular or debug)
#SBATCH -J base_mae_pretrain_job     # Job name
#SBATCH --mail-user=shuklashashankshekhar863@gmail.com  # Email notifications
#SBATCH --mail-type=ALL     # Notifications on job state
#SBATCH -t 00:29:00         # Max runtime
#SBATCH -A m4392         # Allocation account


export HDF5_USE_FILE_LOCKING=FALSE

module load pytorch/2.1.0-cu12
nvidia-smi

srun torchrun --standalone --nproc_per_node=2 /global/homes/s/ssshukla/Shashank/scripts/Reconstruction/model/train.py \
                --epochs=1 \
                --batch_size=256 \
                --model_name="conv_mae" \
                --learning_rate=0.00015 \
                --train_samples=20000 \
                --patch_size=4 \
                --embed_dim=64 \
                --num_heads=8 \
                --decoder_embed_dim=256 \
                --decoder_depth=8 \
                --data_path='/pscratch/sd/s/ssshukla/Boosted_Top.h5' \
                --warmup=3




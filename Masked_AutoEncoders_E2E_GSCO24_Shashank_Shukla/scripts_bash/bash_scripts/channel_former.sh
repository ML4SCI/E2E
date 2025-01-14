#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -n 1                # Number of tasks
#SBATCH -G 4                # Request 2 GPUs
#SBATCH -C gpu&hbm80g       # Ensure we get GPU nodes
#SBATCH --cpus-per-task=64
#SBATCH -q regular        # Queue (e.g., regular or debug)
#SBATCH -J pretrain_job_channel_former     # Job name
#SBATCH --mail-user=shuklashashankshekhar863@gmail.com  # Email notifications
#SBATCH --mail-type=ALL     # Notifications on job state
#SBATCH -t 47:59:00         # Max runtime
#SBATCH -A m4392         # Allocation account


export HDF5_USE_FILE_LOCKING=FALSE

module load pytorch/2.1.0-cu12
nvidia-smi

srun torchrun --standalone --nproc_per_node=4 /global/homes/s/ssshukla/Shashank/scripts/Reconstruction/model/train.py \
                --epochs=100 \
                --batch_size=512 \
                --model_name="channel_former" \
                --learning_rate=0.00015 \
                --train_samples=2000000 \
                --data_path='/pscratch/sd/s/ssshukla/Boosted_Top.h5' \
                --warmup=3 \




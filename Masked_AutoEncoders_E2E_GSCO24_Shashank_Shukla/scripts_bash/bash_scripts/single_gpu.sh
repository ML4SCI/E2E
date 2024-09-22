#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -n 1                # Number of tasks (one per node, mp.spawn will handle multiple processes for GPUs)
#SBATCH -G 1                # Request 2 GPUs
#SBATCH -C gpu              # Ensure we get GPU nodes
#SBATCH -q regular          # Queue (e.g., regular or debug)
#SBATCH -J pretrain_job     # Job name
#SBATCH --mail-user=shuklashashankshekhar863@gmail.com  # Email notifications
#SBATCH --mail-type=ALL     # Notifications on job state
#SBATCH -t 01:00:00         # Max runtime
#SBATCH -A m4392         # Allocation account

export NCCL_DEBUG=INFO

module load pytorch/2.1.0-cu12
nvidia-smi

srun python3 /global/homes/s/ssshukla/scripts/single_gpu.py --epochs=1 --batch_size=256

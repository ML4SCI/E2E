#!/bin/bash
#SBATCH --job-name=pytorch_distributed_training  # Job name
#SBATCH --output=logs/%x_%j.out                 # Standard output and error log
#SBATCH --mail-user=lokeshbadisa657@gmail.com
#SBATCH --mail-type=ALL

#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=0
#SBATCH --gpus-per-task=4
#SBATCH --requeue
#SBATCH --mem=60G

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export SLURM_CPU_BIND="cores"

module load python
conda activate lokesh    

srun python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base


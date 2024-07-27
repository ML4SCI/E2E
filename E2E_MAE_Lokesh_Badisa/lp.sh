#!/bin/bash
#SBATCH --job-name=linear_probing  # Job name
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


# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export SLURM_CPU_BIND="cores"

module load python
conda activate lokesh    

srun python3 dist-lp.py --weights ./Pretraining/vitmae/vit_base/best.pt --runname vit_base --blr 0.1 --config base

salloc -A m4392 -C gpu -q interactive -t 00:30:00 -N 4 --ntasks-per-node=1 -c 4 --mem=0 --gpus-per-task=4 srun python3 dist-lp.py --weights ./Pretraining/vitmae/vit_base/best.pt --runname vit_base --blr 0.1 --config base
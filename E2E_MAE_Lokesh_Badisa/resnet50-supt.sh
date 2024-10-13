#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu 16
#SBATCH --image=docker:ereinha/ngc-24.05-with-addons:latest
#SBATCH --output=logs/%x_%j.out  
#SBATCH --error=errors/%x-%j.out

module load python
conda activate lokesh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1


srun --export=ALL shifter python3 dist-sup-training.py --runname after_deadline_qg --model resnet18\
 --epochs 100 --warmup_epochs 15 --batch_size 256 --optim adam --blr 2e-4

srun --export=ALL shifter python3 dist-sup-training.py --runname after_deadline_bt --model resnet18\
 --data_dir /global/cfs/cdirs/m4392/ACAT_Backup/Data/Top/Boosted_Top.h5 --batch_size 256 --blr 2e-4 \
 --epochs 100 --warmup_epochs 15 --optim adam
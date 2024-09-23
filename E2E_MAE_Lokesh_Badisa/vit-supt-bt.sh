#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu 16
#SBATCH --image=docker:ereinha/ngc-24.05-with-addons:latest
#SBATCH --output=logs/%x_%j.out  
#SBATCH --error=errors/%x-%j.out


export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1
srun --export=ALL shifter python3 dist-sup-training.py --runname vit_supt_100epoch_BT_noaug --model vit_base --batch_size 256 \
 --data_dir /global/cfs/cdirs/m4392/ACAT_Backup/Data/Top/Boosted_Top.h5 --epochs 100 --warmup_epochs 15 --blr 1e-4
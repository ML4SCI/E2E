#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 4
#SBATCH --gpus 4
#SBATCH --cpus-per-gpu 16
#SBATCH --image=docker:ereinha/ngc-24.05-with-addons:latest
#SBATCH --output=logs/%x_%j.out  
#SBATCH --error=errors/%x-%j.out


nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
srun --unbuffered --export=ALL shifter python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base --epochs 1
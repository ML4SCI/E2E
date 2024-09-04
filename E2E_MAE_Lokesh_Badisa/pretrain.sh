#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -N 2
#SBATCH -q preempt
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu 16
#SBATCH --image=docker:ereinha/ngc-24.05-with-addons:latest
#SBATCH --output=logs/%x_%j.out  
#SBATCH --error=errors/%x-%j.out


nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1
srun --export=ALL shifter python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base
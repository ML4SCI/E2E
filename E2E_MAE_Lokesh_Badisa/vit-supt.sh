#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 24:00:00
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
srun --export=ALL shifter python3 dist-sup-training.py --runname vit_supt_100epoch_declr --model vit_base --batch_size 128\
 --blr 1e-5
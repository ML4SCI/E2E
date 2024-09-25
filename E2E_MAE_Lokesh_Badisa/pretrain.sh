#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 30:00:00
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

srun --export=ALL shifter python3 dist-training.py --runname after_deadline_qg --config small\
 --epochs 100 --warmup_epochs 15 --blr 2e-4

srun --export=ALL shifter python3 dist-sup-training.py --runname mae_ft_qg --model vit_small\
 --epochs 20 --warmup_epochs 5 --batch_size 128 --optim adamw --blr 1.5e-4 --weights \
 ./Pretraining/vitmae/after_deadline_qg/best.pt

srun --export=ALL shifter python3 dist-sup-training.py --runname mae_lp_qg --model vit_small\
 --epochs 20 --warmup_epochs 5 --batch_size 128 --optim adamw --blr 1.5e-4 --weights \
 ./Pretraining/vitmae/after_deadline_qg/best.pt --lp
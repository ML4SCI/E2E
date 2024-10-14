#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 04:00:00
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
srun --export=ALL shifter python3 dist-sup-training.py --runname mae_ft --model vit_base\
 --epochs 10 --warmup_epochs 3 --batch_size 128 --optim adamw --blr 1.5e-4 --weights \
 ./Pretraining/vitmae/on_deadline_qg/best.pt


salloc -A m4392 -C gpu -N 1 -q interactive -t 04:00:00 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-gpu=16 python3 mod-training.py --runname mae_ft --model vit_base --epochs 10 --warmup_epochs 3 --batch_size 128 --optim adamw --blr 1.5e-4 --weights ./Pretraining/vitmae/on_deadline_qg/best.pt
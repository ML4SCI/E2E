#!/bin/bash
#SBATCH --job-name=pytorch_distributed_training  # Job name
#SBATCH --output=logs/%x_%j.out                 # Standard output and error log


#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --requeue

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --export=ALL podman-hpc run -v \
/global/homes/l/lokeshb/E2E/E2E_MAE_Lokesh_Badisa:/workspace --rm lokesh:v5 \
python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base --epochs 1
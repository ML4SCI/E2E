#!/bin/bash
#SBATCH --job-name=pytorch_distributed_training  # Job name
#SBATCH --output=logs/%x_%j.out                 # Standard output and error log
#SBATCH --mail-user=lokeshbadisa657@gmail.com
#SBATCH --mail-type=ALL

#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gpus-per-task=4
#SBATCH --requeue
#SBATCH --mem=60G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

module load python
conda activate lokesh    

srun torchrun --nnodes 2 --nproc_per_node 4 \
--rdzv_id $RANDOM --rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \ 
python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base

#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -n 1                # Number of tasks (one per node, mp.spawn will handle multiple processes for GPUs)
#SBATCH -G 2                # Request 2 GPUs
#SBATCH -C gpu              # Ensure we get GPU nodes
#SBATCH --cpus-per-task=32  # Number of cpus required
#SBATCH -q regular          # Queue (e.g., regular or debug)
#SBATCH -J pretrain_job     # Job name
#SBATCH --mail-user=shuklashashankshekhar863@gmail.com  # Email notifications
#SBATCH --mail-type=ALL     # Notifications on job state
#SBATCH -t 01:00:00         # Max runtime
#SBATCH -A m4392         # Allocation account


# nvidia-smi

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export MASTER_ADDR=$(hostname)
export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam
# TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
# export TIMESTAMP

srun --unbuffered --export=ALL shifter python3 new_method_train.py --WandB --run_test --model_name=Depthwise_conv --epochs=1 --base_lr=.00004 --epsilon=0.01 --batch_size=512 --warmup=25 --num_worker=8 --checkpoint_folder=/global/homes/s/ssshukla/scripts/Results/Weights/depthwise_conv/ --n_train=-1
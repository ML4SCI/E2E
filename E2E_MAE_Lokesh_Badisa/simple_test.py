from torch.distributed import init_process_group
import os

def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir,exist_ok=True)
        sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        return sync_file

sync_file = _get_sync_file()

def ddp_setup():
    WORLD_SIZE = int(os.environ['SLURM_NTASKS'])  # number of nodes
    GLOBAL_RANK = int(os.environ['SLURM_PROCID'])
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend="nccl", world_size=WORLD_SIZE, rank=GLOBAL_RANK, init_method=sync_file)

WORLD_SIZE = int(os.environ['SLURM_NTASKS'])  # number of nodes
GLOBAL_RANK = int(os.environ['SLURM_PROCID'])
LOCAL_RANK = int(os.environ['SLURM_LOCALID'])


print(f"WORLD_SIZE: {WORLD_SIZE} GLOBAL_RANK: {GLOBAL_RANK} LOCAL_RANK: {LOCAL_RANK}")
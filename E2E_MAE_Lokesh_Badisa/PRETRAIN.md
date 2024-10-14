To run the code in interactive mode:
```
sbatch pretrain.sh
```

salloc -N 1 -c 4 --gpus-per-task=1 --time=00:30:00 --qos=interactive --account=m4392 -C gpu --ntasks-per-node=4 srun python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config base
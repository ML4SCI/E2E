To run the code in interactive mode:
```
salloc -N 4 -c 4 --gpus-per-task=4 --time=00:30:00 --qos=interactive --a
ccount=m4392 -C gpu --ntasks-per-node=1 srun python3 dist-training.py --runname vit_base --blr 1.5e-4 --mask 0.75 --config ba
se
```
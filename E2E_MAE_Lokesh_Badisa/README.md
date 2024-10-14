# Masked Auto-Encoders for Efficient End-to-End Particle Reconstruction and Compression for the CMS Experiment

This repository contains the code for the paper "Masked Auto-Encoders for Efficient End-to-End Particle Reconstruction and Compression for the CMS Experiment" by the CMS Collaboration. Follow this [blog](https://medium.com/@lokeshbadisa657/gsoc-2024-with-ml4sci-masked-auto-encoders-for-efficient-end-to-end-particle-reconstruction-and-60ea4dde539e) to read about the work done in this project.

The code is suitable to run on a SLURM cluster. Please change the slurm variables in the scripts to run with different gpu setting. We use multi-node training to decrease the training time(if you've been searching for multi-node training😉). [Official Google Page](https://summerofcode.withgoogle.com/programs/2024/projects/4KpQiVr8) for the project.

## Installation
To install the required packages, run the following command:
```
pip install -r requirements.txt
```

## Training ViT-MAE
To pretrain, linear-probe & finetune the model on boosted-top dataset, run the following command:
```
sbatch pretrain-bt.sh
```

To pretrain, linear-probe & finetune the model on quark-gluon dataset, run the following command:
```
sbatch pretrain.sh
```

## Training ViT in supervised fashion
To train ViT in supervised fashion on Quark-Gluon dataset, run the following command:
```
sbatch vit-supt.sh
```
Parallely for Boosted-Top dataset, run the following command:
```
sbatch vit-supt-bt.sh
```

## Training resnet50 in supervised fashion
To train ViT in supervised fashion on Quark-Gluon dataset, run the following command:
```
sbatch resnet50-supt.sh
```
Parallely for Boosted-Top dataset, run the following command:
```
sbatch resnet50-supt-bt.sh
```

* Please see `dist-training.py` for hackable hyper-parameters in training ViT-MAE.
* Please see `dist-sup-training.py` for hackable hyper-parameters in training supervised models(both ViT & resnet50).

## Acknowledgements
I would like to thank Eric Reinhardt, Diptarko Choudhary, Dr. Sergei Gleyzer and E2E team for their technical support.

### References
He, Kaiming, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. “Masked autoencoders are scalable vision learners.” In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009. 2022.
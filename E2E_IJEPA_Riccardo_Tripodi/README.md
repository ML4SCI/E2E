# E2E - Self-Supervised Learning for End-to-End Particle Reconstruction for the CMS Experiment

![Alt text](https://github.com/3podi/ijepa_gsoc/blob/main/imgs/intro_img.jpg)

In this repository it can be found the work done under Machine Learning for Science (ML4Sci) as a Google Summer of Code (GSoC) 24' contributor. ML4Sci is an open-source organization that brings together modern machine learning techniques and applies them to cutting edge STEM problems and GSoC is a global, online program focused on bringing new contributors into open source software development.

## Project aim
The aim of the project is to investigate on the use of an I-JEPA architecture for the self-supervised pre-traning stage on unlabeled data from the CMS experiment. The pre-training stage is validated fine-tuning by linear probing the frozen pre-trained models on a downstream binary particle classification task.

To read more about the project:

[Medium blogpost 1/2](https://medium.com/@riccardotripodi/self-supervised-learning-for-end-to-end-particle-reconstruction-for-the-cms-experiment-1-2-6d4d14e8c45b)

[Medium blogpost 2/2](https://medium.com/@riccardotripodi/self-supervised-learning-for-end-to-end-particle-reconstruction-for-the-cms-experiment-2-2-9997aa51ca7d)

## Code
You can run the following line of code to start the training of a model with a given configuration:
```bash
python main.py --fname /configs/vit_b_14.yaml devices cuda:0
```
Moreover, specifying the number of layers to unfreeze in the config file, you can linear probe or fine tune a model with the line:
```bash
python linear_probing.py  --fname configs/probing_vit_b_14.yaml --devices cuda:0
```
## Results
Those are the obtained results. All scores are ROC-AUC metric. To read more about those results check out the second blogpost.

| Model Name      |  Scratch        | Linear probing | Fine-tuning        |
| --------------- | --------------- | -------------- | ------------------ |
| vit_s_14        | 0.75            | 0.732          | 0.74               |
| vit_s_9         | 0.76            | 0.731          | 0.76               |
| vit_b_14        | 0.74            | 0.737          | 0.78               |
| vit_b_9         | 0.73            | 0.738          | 0.79               |

All the pre-trained models can be downloaded at the following google drive link:

[Download models]()

## Playground
You can pre-train a 'vit_base' encoder and finetune it on a binary classification task on colab using the following colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/ijepa_gsoc/blob/main/notebooks/train_notebook.ipynb)

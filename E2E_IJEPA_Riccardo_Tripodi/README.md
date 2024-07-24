# E2E - Self-Supervised Learning for End-to-End Particle Reconstruction for the CMS Experiment

![Alt text](https://github.com/3podi/ijepa_gsoc/blob/main/imgs/intro_img.jpg)

In this repository it can be found the work done under Machine Learning for Science (ML4Sci) as a Google Summer of Code (GSoC) 24' contributor. ML4Sci is an open-source organization that brings together modern machine learning techniques and applies them to cutting edge STEM problems and GSoC is a global, online program focused on bringing new contributors into open source software development.

## Project aim
The aim of the project is to investigate on the use of an I-JEPA architecture for the self-supervised pre-traning stage on unlabeled data from the CMS experiment. The pre-training stage is validated fine-tuning by linear probing the frozen pre-trained model on a downstream binary classification task.

To read more about the project:
[Medium blogpost 1/2](https://medium.com/@riccardotripodi/self-supervised-learning-for-end-to-end-particle-reconstruction-for-the-cms-experiment-1-2-6d4d14e8c45b)

## Code
To run the project, after cloning this repo and installing the required libraries, you can run the following line of code:
```bash
python main.py --fname /configs/vit_b_14.yaml devices cuda:0
```

## Playground
You can pre-train a 'vit_base' encoder and finetune it on a binary classification task on colab using the following colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/ijepa_gsoc/blob/main/notebooks/train_notebook.ipynb)

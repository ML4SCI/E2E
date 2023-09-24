# Vision Transformers for End-to-End Particle Reconstruction for the CMS Experiment
This repository contains the code for the Google Summer of Code (GSoC) 2023 project: Vision Transformers (ViT) for End-to-End Particle Reconstruction for the CMS Experiment. The project is primarily written with python. Details of the project is summarized in a two-part blog post: [part 1](https://medium.com/@kho2_97849/vision-transformers-for-end-to-end-particle-reconstruction-at-the-cms-experiment-75d5758a98f3) and [part 2](https://medium.com/@kho2_97849/vision-transformers-for-end-to-end-particle-reconstruction-at-the-cms-experiment-part-2-2-5773525466df).

# Setup
To install all the required package, do:

```bash
pip install -r requirements.txt
```

# Download and preprocess the dataset

## Quark-Gluon dataset

To obtain the quark-gluon dataset [1], please email the ML4SCI organisation at ml4-sci@cern.ch. After downloading the quark-gluon dataset, create the following folders
```bash
mkdir -p ./data/QG/parquet/train
mkdir -p ./data/QG/parquet/test
```
, and place the training (testing) dataset in ```./data/QG/parquet/train``` (```./data/QG/parquet/test```). The raw images have then to be preprocessed. For the training set,
1. pixels with values 0.001 are set to zero
2. images are normalized to have a mean of 0 and a standard deviation of 1 per channel in batches of 4096
3. pixels with values that are 500 standard deviation away from the mean per channel are set to zero within each image.

For the testing set, similar preprocessing was carried out except the images are normalized according to the global mean and standard deviation per channel of the training set instead of a batched mean/standard deviation. To preprocess the dataset, do
```bash
python preprocessor/QG/preprocessor.py
```
, which will create two PyTorch files ```data/QG/tensor/train.pt``` and ```data/QG/tensor/test.pt``` storing the images in tensor ready for model training and testing.

## Boosted Top dataset

Similarly, to download the boosted top dataset [2], please email the ML4SCI organisation at ml4-sci@cern.ch. After downloading the dataset, place the parquet files in ```./data/top/parquet```. The raw images have to be preprocessed in a similar way to the quark-gluon dataset. To preprocess the dataset, do
```bash
python preprocessor/top/preprocessor.py
```
, which will process the files to the TFRecord format ready for training/validation/testing!

# Training and testing
This section uses the wandb package for logging. To setup an account, please visit https://wandb.ai.
## Swin transformer [3]
To start the training and testing process of the quark-gluon dataset, do
```bash
python run_train_test_parallel.py -sw
```
The ```-sw``` flag here turns on the shifting window operation in Swin. User could modify the patch size with the flag ```-p``` followed by an integer, the window size with the flag ```-w``` followed by an integer, the embedding dimension with the flag ```-e``` followed by an integer, and the number of heads in each layer with the flag ```-h``` followed by comma separated integers. For the number of layers, it is fixed to be the same as Swin-Tiny as in [2]. The user could also modifiy the number of gpu used for the training/testing with the flag ```-gpu```.

Similary, to start the training and testing process of the boosted top dataset, do
```bash
python run_train_test_TPU.py -sw
```
It makes use of the TPU cloud provided by Google. To see more details, please visit https://cloud.google.com/tpu. Same sets of input flags are available for the ```run_train_test_TPU.py``` script with the exception of the ```-gpu``` flag.

## Multi-scale Swin
Multi-scale Swin (MSwin) employs various window sizes in the training by dividing the hidden dimension of the image equally for each window size. In this case, the various heads in the multi-head attention (MHA) mechansim would attend to a different window size. To start the training and testing process, do
```bash
python run_train_test_parallel.py -sw -w 4,8,16
```
The ```-w``` specifies that a model with window sizes of 4x4, 8x8, and 16x16 will be used. Similarly, the ```run_train_test_TPU.py``` can be used for the boosted top dataset with the same option flags.

## Win [4]
To start the training and testing process, do
```bash
python run_train_test_parallel.py -cw
```
The ```-cw``` flag and the absent of the ```-sw``` flag specify that a layer-wise convolution will be used in placed of the shifting window operation. Similar to SWin and MSWin, user could also alter the hyperparameters, like the hidden dimension, the window sizes or train with multi-scale windows. Similarly, the ```run_train_test_TPU.py``` can be used for the boosted top dataset with the same option flags.

# Acknowledgment
Many thanks to Google for providing the free TPU Cloud service to make training of the boosted top dataset possible.

Also, thanks to Dr. Sergei Gleyzer for his guidance during the project and Diptarko Choudhury for his valuable insights.

# References
[1]: Andrews, Michael, et al. “End-to-end jet classification of quarks and gluons with the CMS Open Data.” Nuclear instruments and methods in physics research section A: accelerators, spectrometers, detectors and associated equipment 977 (2020): 164304.

[2]: Andrews, Michael, et al. "End-to-end jet classification of boosted top quarks with the CMS open data." Physical Review D 105.5 (2022): 052008.

[3]: Liu, Ze, et al. “Swin transformer: Hierarchical vision transformer using shifted windows.” Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[4]: Yu, Tan, and Ping Li. “Degenerate Swin to win: Plain window-based transformer without sophisticated operations.” arXiv preprint arXiv:2211.14255 (2022).

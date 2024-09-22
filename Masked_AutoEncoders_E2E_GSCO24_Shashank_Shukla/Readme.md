# Specific Task 3d: Masked Auto-Encoder for Efficient End-to-End Particle Reconstruction and Compression

### Tasks
1. Train a lightweight ViT using the Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
2. Compare reconstruction results using MAE on both training and testing datasets.
3. Fine-tune the model on a lower learning rate on the provided labelled dataset and compare results with a model trained from scratch.

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/MAE.png" width="700" title="hover text">
</p>

### Implementation
- Trained a lightweight ViT using MAE on unlabelled dataset
- Compared reconstruction results on training and testing datasets
- Fine-tuned the model on a lower learning rate using the labelled dataset
- Compared results with a model trained from scratch
- Ensured no overfitting on the test dataset

### Image Reconstruction
####                                           Original
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Original.jpg" width="700" title="hover text">
</p>

####                                           Reconstructed
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Reconstructed.jpg" width="700" title="hover text">
</p>

### Comparison of Different Masked AutoEncoder Architectures
##                                            Linear Probing
<p align="center">
  <img src="https://github.com/Wodlfvllf/E2E/blob/main/Masked_AutoEncoders_E2E_GSCO24_Shashank_Shukla/Linear%20Probing%20Table.png" width="700" title="hover text">
</p>       

##                                            FineTuning
<p align="center">
  <img src="https://github.com/Wodlfvllf/E2E/blob/main/Masked_AutoEncoders_E2E_GSCO24_Shashank_Shukla/Finetuning%20Table.png" width="700" title="hover text">
</p>      

## Refer to this blog for details of the project.
[Masked Auto-Encoders](https://medium.com/@shuklashashankshekhar863/masked-autoencoders-for-efficient-end-to-end-particle-reconstruction-and-compression-for-the-cms-fdd7b941a2bb)
## Dependencies
- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install these dependencies using pip or conda.

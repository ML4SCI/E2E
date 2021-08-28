import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import *
import random
import matplotlib.pyplot as plt
import json
from data_utils import train_val_loader
from models import ResNet
from train_utils import train

plt.rcParams["figure.figsize"] = (5, 5)
plt.switch_backend('agg')

params = json.load(open("./E2E/E2E_Regression_Anis_Ismail/experiment.json",'r'))
np.random.seed(params["seed"])
torch.manual_seed(params["seed"])
random.seed(params["seed"])

LOG_PATH = os.path.join(params["save_path"], "LOGS")
MODEL_PATH = os.path.join(params["save_path"], "MODELS")
PLOT_PATH = os.path.join(params["save_path"], "PLOTS")

if not os.path.isdir(LOG_PATH):
    os.makedirs(LOG_PATH)
for d in [MODEL_PATH, PLOT_PATH]:
    if not os.path.isdir('%s/%s' % (d, params["expt_name"])):
        os.makedirs('%s/%s' % (d, params["expt_name"]))

train_loader, val_loader, test_loader = train_val_loader(
    [os.path.join(params["data_path"], f) for f in
     os.listdir(params["data_path"])], params["batch_size"])

resnet = ResNet(params["input_channels"], params["resblocks"], params["fmaps"])
if params["is_cuda"]:
    resnet.cuda()
optimizer = optim.Adam(resnet.parameters(), lr=params["lr"])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
train(resnet, optimizer, lr_scheduler, params["epochs"], train_loader, val_loader)

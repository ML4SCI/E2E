import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, models, transforms

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc
import wandb
from coat import *


def metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc

def straightner(a):
    A = np.zeros((a[0].shape[0]*len(a)))
    start_index = 0
    end_index = 0
    for i in range(len(a)):
        start_index = i*a[0].shape[0]
        end_index = start_index+a[0].shape[0]
        A[start_index:end_index] = a[i]
    return A

def predictor(outputs):
    return np.argmax(outputs, axis = 1)


def trainer(num):



    image_size = (128,128)
    in_channels = 3
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    num_classes = 1

    #-------------------------------------------------
    model = CoAtNet(image_size = image_size,
                         in_channels = in_channels,
                         num_blocks = num_blocks,
                         channels = channels,
                         num_classes = num_classes)
    #---------------------------------------------------
    train_transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(60),
                                transforms.ToTensor()
                               ])
    test_transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor()
                               ])
    dataset_Train = datasets.ImageFolder(f'./Data/Train/', transform=train_transform)
    dataset_Test = datasets.ImageFolder(f'./Data/Test/', transform =test_transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_Train, batch_size=256, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)
    dataloader_test = torch.utils.data.DataLoader(dataset_Test, batch_size=256, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)
    #--------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 3, factor = 0.5)

    model = model.to("cuda")
    #---------------------------------------------------------


    wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
    wandb.init(
         project = "Ensemble",
         name = f"CoAt_{num}"
         )
    #-----------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler()
    #--------------------------
    wandb.watch(model, log_freq=50)
    #---------------------------
    w_intr = 50

    for epoch in range(100):
        train_loss = 0
        val_loss = 0
        train_steps = 0
        test_steps = 0
        label_list = []
        outputs_list = []
        train_auc = 0
        test_auc = 0
        model.train()
        for image, label in tqdm(dataloader_train):
            image = image.to("cuda")
            label = label.to("cuda")
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
              outputs = model(image)
              loss = criterion(outputs, label.float())

            label_list.append(label.detach().cpu().numpy())
            outputs_list.append(outputs.detach().cpu().numpy())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_steps += 1
            if train_steps%w_intr == 0:
                 wandb.log({"loss": loss.item()})
        with torch.no_grad():
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            train_auc = metric(label_list, outputs_list)




        #-------------------------------------------------------------------
        model.eval()
        label_list = []
        outputs_list = []
        with torch.no_grad():
            for image, label in tqdm(dataloader_test):
                image = image.to("cuda")
                label = label.to("cuda")
                outputs = model(image)
                loss = criterion(outputs, label.float())
                label_list.append(label.detach().cpu().numpy())
                outputs_list.append(outputs.detach().cpu().numpy())
                val_loss += loss.item()
                test_steps +=1
                if test_steps%w_intr == 0:
                 wandb.log({"val_loss": loss.item()})
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            test_auc = metric(label_list, outputs_list)

        train_loss = train_loss/train_steps
        val_loss = val_loss/ test_steps


        print("----------------------------------------------------")
        print("Epoch No" , epoch)
        print("The Training loss of the epoch, ",train_loss)
        print("The Training AUC of the epoch,  %.3f"%train_auc)
        print("The validation loss of the epoch, ",val_loss)
        print("The validation AUC of the epoch, %.3f"%test_auc)
        print("----------------------------------------------------")
        PATH = f"model_epoch_{epoch}_run_{num}.pt"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, PATH)
        scheduler.step(test_auc)
        curr_lr = scheduler._last_lr[0]
        wandb.log({"Train_auc_epoch": train_auc,
                  "Epoch": epoch,
                  "Val_auc_epoch": test_auc,
                  "Train_loss_epoch": train_loss,
                  "Val_loss_epoch": val_loss,
                  "Lr": curr_lr}
                 )
        gc.collect()

        if curr_lr < 0.000001:
            break
    wandb.finish()


if __name__ == "__main__":
    trainer(0)

import timm
import torch
import numpy as np
import torch.nn as nn
import timm
from torchvision import datasets, models, transforms

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc 
#Stage 1> A clean Swin transformer model will be trained trained
#Stage 2> The swin Transformer blocks are frozen and the new embeding layer is attached and trained
#The old embedding blocks are replaced by the CNNs.In this stage only the embeding layer is trained
#Stage 3> The entire model is unfrozen and trained

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

class Hybrid_embed(nn.Module):
    def __init__(self, feature_model, img_size, channels, efn_blocks, dims):
        super().__init__()
        
        
        self.feature_extractor = timm.create_model(feature_model,
                                                   features_only=True,
                                                   out_indices=[efn_blocks])
        
        
        self.feature_extractor.conv_stem = nn.Conv2d(3,   
                                       40,
                                       kernel_size=(3, 3),
                                       stride=(4, 4),
                                       padding=(1, 1),
                                       bias=False)
        
        with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = self.feature_extractor.training
                if training:
                    self.feature_extractor.eval()
                o = self.feature_extractor(torch.zeros(1, channels, img_size[0], img_size[1]))
                self.channel_output = o[0].shape[1]
                self.feature_extractor.train(training)
        
        self.embed_matcher = nn.Sequential(
            nn.Conv2d(self.channel_output, dims, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dims, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )
        
        
    def forward(self, x):
        x = self.feature_extractor(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.embed_matcher(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Hybrid_swin_effnet(nn.Module):
    def __init__(self, feature_model = "efficientnet_b3",img_size = (224,224), channels = 3, efn_blocks = 2, swin_blocks = 2, no_classes = 1):
        super().__init__()
        assert efn_blocks + swin_blocks == 4,f"The total no of blocks must be 4, instead {efn_blocks+swin_blocks} blocks provided "
        self.s1_flag = True
        self.s2_flag = True
        self.s3_flag = True
        self.swin_blocks = swin_blocks
#         self.feature_extractor = timm.create_model(feature_model,
#                                                    features_only=True,
#                                                    out_indices=[efn_blocks])
        
#         #Removing the initial stem layer since our image size is pretty low and we have already upscaled it.
#         self.feature_extractor.conv_stem = nn.Conv2d(3,   
#                                        40,
#                                        kernel_size=(3, 3),
#                                        stride=(4, 4),
#                                        padding=(1, 1),
#                                        bias=False)
        
#         #------------------------------------------------------------------
#         with torch.no_grad():
#                 # NOTE Most reliable way of determining output dims is to run forward pass
#                 training = self.feature_extractor.training
#                 if training:
#                     self.feature_extractor.eval()
#                 o = self.feature_extractor(torch.zeros(1, channels, img_size[0], img_size[1]))
#                 self.channel_output = o[0].shape[1]
#                 self.feature_extractor.train(training)
        
#         #------------------------------------------------------------------
        self.swin_backbone = timm.create_model("swin_tiny_patch4_window7_224")
        
#         self.original_embed = self.swin_backbone.patch_embed
        
        self.embeded_dim = self.swin_backbone.embed_dim * (2**(4 - self.swin_blocks))
        
#         self.embed_matcher = nn.Sequential(
#             nn.Conv2d(self.channel_output, self.embeded_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
#             nn.BatchNorm2d(self.embeded_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.SiLU(inplace=True)
#         )
        
#         #Matching the output channel
#         self.patc_embed_hybrid = nn.Sequential(self.feature_extractor, self.embed_matcher)
#         # !!!!!!!! The [0] should be inspected !!!!!
        
#         self.swin_backbone.patch_embed = self.patc_embed_hybrid


        self.Hybrid_patch_embed = Hybrid_embed(feature_model = "efficientnet_b3",
                                                      img_size = (224,224),
                                                      channels = 3,
                                                      efn_blocks = 2, 
                                                      dims = self.embeded_dim)
        
        #setting the first few blocks of swin to Indentity to match size
#         for i in range((4- swin_blocks)):
#             self.swin_backbone.layers[i] = nn.Identity()
        
        #Setting the head as per our need
        self.swin_backbone.head = nn.Linear(self.swin_backbone.num_features, no_classes)
        
    def forward(self, image, stage):
        
        
        if stage == 2:
            #Attaching the new embeding layer
                
            self.swin_backbone.patch_embed = self.Hybrid_patch_embed

            for i in range((4- self.swin_blocks)):
                self.swin_backbone.layers[i] = nn.Identity()

            #Freezing the swin layers    
            for layer in self.swin_backbone.layers:
                for para in layer.parameters():
                    para.requires_grad = False

           #Freezing the head    
            for para in self.swin_backbone.head.parameters():
                    para.requires_grad = False
            
        if stage == 3:
            #Unfreezing the network
            for para in self.swin_backbone.parameters():
                para.requires_grad = True
    
                
        return self.swin_backbone(image).squeeze()
    
def main():
    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor()
                            ])
    test_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()
                            ])





    criterion = nn.BCEWithLogitsLoss()
    model = Hybrid_swin_effnet()
    model = model.to("cuda")



    import wandb
    wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
    wandb.init(
        project = "Ensemble",
        name = "Eff2_Swin_trained_step_by_step_rtc"
    )

    sample = torch.randn(1, 3, 224, 224, device = "cuda")

    scaler = torch.cuda.amp.GradScaler()
    #--------------------------
    wandb.watch(model, log_freq=50)
    #---------------------------
    w_intr = 50

    dataset_Train = datasets.ImageFolder('./Data/Train/', transform=train_transform)
    dataset_Test = datasets.ImageFolder('./Data/Test/', transform =test_transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_Train, batch_size=100, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)
    dataloader_test = torch.utils.data.DataLoader(dataset_Test, batch_size=100, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)    

    print("Entering stage 1")
    #Stage 1 -----------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 3, factor = 0.5)

    for epoch in range(50):
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
                outputs = model(image ,1)
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
                outputs = model(image ,1)
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
        PATH = f"model_stage_1_{epoch}.pt"
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
                "Lr": curr_lr, 
                "Stage": 1
                }
                )
        gc.collect()
        

    dataset_Train = datasets.ImageFolder('./Data/Train/', transform=train_transform)
    dataset_Test = datasets.ImageFolder('./Data/Test/', transform =test_transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_Train, batch_size=200, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)
    dataloader_test = torch.utils.data.DataLoader(dataset_Test, batch_size=200, shuffle=True, drop_last = True, num_workers=4, pin_memory = True)    


    #Stage 2--------------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 3, factor = 0.5)

    with torch.no_grad():
        model(sample,2)

    for epoch in range(50):
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
                outputs = model(image ,1)
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
                outputs = model(image, 1)
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
        PATH = f"model_stage_2_{epoch}.pt"
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
                "Lr": curr_lr, 
                "Stage": 2
                }
                )
        gc.collect()
        
    #Stage 3--------------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 5, factor = 0.7)

    with torch.no_grad():
        model(sample,3)

    for epoch in range(150):
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
                outputs = model(image ,1)
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
                outputs = model(image,1)
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
        PATH = PATH = f"model_stage_3_{epoch}.pt"
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
                "Lr": curr_lr, 
                "Stage": 3
                }
                )
        gc.collect()
        
if __name__ == "__main__":
    main()

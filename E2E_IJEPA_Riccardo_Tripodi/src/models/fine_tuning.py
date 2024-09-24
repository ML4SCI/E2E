import numpy as np
import torch
import torch.nn as nn

from src.models.resnet import ResNet50
import src.models.vision_transformer as vit


class LinearProbe(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes, use_batch_norm=False, use_hidden_layer=False, num_unfreeze_layers=0):
        super().__init__()
        self.encoder = pretrained_model

        # Freeze the encoder's parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        if num_unfreeze_layers > 0:
            for block in self.encoder.blocks[-num_unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        if use_hidden_layer:
            self.linear = nn.Sequential(
                nn.Linear(hidden_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes)
            )
        else:
            self.linear = nn.Linear(hidden_dim,num_classes)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()


    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling

        x = self.batch_norm(x)
        return self.linear(x)
        
class AddLinear(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, use_batch_norm=False, use_hidden_layer=False):
        super().__init__()
        self.encoder = encoder

        if use_hidden_layer:
            self.linear = nn.Sequential(
                nn.Linear(hidden_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes)
            )
        else:
            self.linear = nn.Linear(hidden_dim,num_classes)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()
    
    def forward(self,x):
        
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.batch_norm(x)

        return self.linear(x)

    

def resnet_50(num_classes, classification_head):
    return ResNet50(num_classes=num_classes,classification_head=classification_head)

def vit_model(num_classes, use_batch_norm, img_size, patch_size,model_name, use_hidden_layer):
    encoder = vit.__dict__[model_name](img_size=[img_size],patch_size=patch_size)
    embed_dim = encoder.embed_dim

    return AddLinear(encoder, embed_dim, num_classes, use_batch_norm, use_hidden_layer)






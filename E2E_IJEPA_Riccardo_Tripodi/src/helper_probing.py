import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.models.fine_tuning import LinearProbe
import src.models.fine_tuning as model_ft



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(
    img_size,
    patch_size,
    pretrained_path=None,
    model_name=None,
    num_classes=1,
    use_batch_norm=False,
    use_hidden_layer=False,
    num_unfreeze_layers=0
):
    
    if pretrained_path is None:
        if 'resnet' in model_name:
            model = model_ft.__dict__[model_name](num_classes,classification_head=True)
        else:
            model = model_ft.__dict__['vit_model'](
                num_classes=num_classes, 
                use_batch_norm=use_batch_norm,
                img_size=img_size,
                patch_size=patch_size,
                model_name=model_name,
                use_hidden_layer=use_hidden_layer)

        return model
    
    else:
        encoder_ijepa = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size)
        embed_dim = encoder_ijepa.embed_dim
    
        checkpoint = torch.load(pretrained_path)
        encoder_ijepa.load_state_dict(checkpoint['target_encoder'])
        model = LinearProbe(encoder_ijepa, embed_dim, num_classes, use_batch_norm, use_hidden_layer, num_unfreeze_layers)
        return model


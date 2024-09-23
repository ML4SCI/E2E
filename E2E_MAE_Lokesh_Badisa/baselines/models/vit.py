from timm.models.vision_transformer import VisionTransformer
from timm.models.crossvit import CrossVit


def vit_tiny(channels: int, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=192, depth=12, num_heads=3, num_classes=2, in_chans=channels, img_size=125, **kwargs)
    return model

def vit_small(channels: int, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=384, depth=12, num_heads=6, num_classes=2, in_chans=channels, img_size=125, **kwargs)
    return model

def vit_base(channels: int, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=768, depth=12, num_heads=12, num_classes=2, in_chans=channels, img_size=125, **kwargs)   
    return model

def vit_large(channels: int, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=1024, depth=24, num_heads=16, num_classes=2, in_chans=channels, img_size=125, **kwargs)
    return model

def crossvit_tiny(channels: int,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[3, 3], mlp_ratio=[4, 4, 1], in_chans=channels, num_classes=2, **kwargs)   
    return model

def crossvit_small(channels: int,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[6, 6], mlp_ratio=[4, 4, 1],  in_chans=channels, num_classes=2, **kwargs)   
    return model

def crossvit_base(channels: int,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[12, 12], mlp_ratio=[4, 4, 1],  in_chans=channels, num_classes=2, **kwargs)   
    return model


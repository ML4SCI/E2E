from timm.models.vision_transformer import VisionTransformer
from timm.models.crossvit import CrossVit
from timm.models.swin_transformer import SwinTransformer


def vit_tiny(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=192, depth=12, num_heads=3, num_classes=2, in_chans=8, img_size=125, **kwargs)
    return model

def vit_small(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=384, depth=12, num_heads=6, num_classes=2, in_chans=8, img_size=125, **kwargs)
    return model

def vit_base(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=768, depth=12, num_heads=12, num_classes=2, in_chans=8, img_size=125, **kwargs)   
    return model

def vit_large(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=5, embed_dim=1024, depth=24, num_heads=16, num_classes=2, in_chans=8, img_size=125, **kwargs)
    return model

def crossvit_tiny(pretrained=False,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[3, 3], mlp_ratio=[4, 4, 1], in_chans=8, num_classes=2, **kwargs)   
    return model

def crossvit_small(pretrained=False,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[6, 6], mlp_ratio=[4, 4, 1],  in_chans=8, num_classes=2, **kwargs)   
    return model

def crossvit_base(pretrained=False,**kwargs):
    model = CrossVit(img_size=125,img_scale=(1.0,125/150),patch_size=(5,25), embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                     num_heads=[12, 12], mlp_ratio=[4, 4, 1],  in_chans=8, num_classes=2, **kwargs)   
    return model

# def swint_tinyw9(pretrained=False,**kwargs):
#     model = SwinTransformer(img_size=125, patch_size=5, in_chans=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), num_classes=2, window_size=9,**kwargs)
#     return model

# def swint_smallw9(pretrained=False,**kwargs):
#     model = SwinTransformer(img_size=125, patch_size=5, in_chans=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), num_classes=2, window_size=9,**kwargs)
#     return model

# def swint_basew9(pretrained=False,**kwargs):
#     model = SwinTransformer(img_size=125, patch_size=5, in_chans=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=2, window_size=9,**kwargs)
#     return model
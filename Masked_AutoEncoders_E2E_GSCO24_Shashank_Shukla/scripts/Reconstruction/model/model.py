from base_mae_depthwise_convolution import *
from base_mae import *
from channel_former import *
from conv_mae import *
from cross_vit import *
from util import *

def get_model(args):
    if args.model_name == "base_mae_depthwise_convolution":
        model = mae_vit_depthwise_conv(img_size=args.img_size, 
                           patch_size=args.patch_size, 
                           in_chans=args.in_chans, 
                           embed_dim=args.embed_dim, 
                           depth=args.depth, 
                           num_heads=args.num_heads,
                           decoder_embed_dim=args.decoder_embed_dim, 
                           decoder_depth=args.decoder_depth, 
                           decoder_num_heads=args.decoder_num_heads, 
                           k_factor = args.k_factor,
                           mlp_ratio=args.mlp_ratio)

    elif args.model_name == "channel_former":
        model = mae_vit_channel_former(img_size=args.img_size, 
                           patch_size=args.patch_size, 
                           in_chans=args.in_chans, 
                           embed_dim=args.embed_dim, 
                           depth=args.depth, 
                           decoder_embed_dim=args.decoder_embed_dim, 
                           decoder_depth=args.decoder_depth, 
                           k_factor = args.k_factor,
                           mlp_ratio=args.mlp_ratio)

    elif args.model_name == "base_mae":
        model = mae_vit_base(img_size=args.img_size, 
                           patch_size=args.patch_size, 
                           in_chans=args.in_chans, 
                           embed_dim=args.embed_dim, 
                           depth=args.depth, 
                           num_heads=args.num_heads,
                           decoder_embed_dim=args.decoder_embed_dim, 
                           decoder_depth=args.decoder_depth, 
                           decoder_num_heads=args.decoder_num_heads, 
                           mlp_ratio=args.mlp_ratio)

    elif args.model_name == "conv_mae":
        model = convmae_convvit_base_patch16_dec512d8b(img_size=args.img_size, 
                           patch_size=args.patch_size, 
                           in_chans=args.in_chans, 
                           embed_dim=args.embed_dim, 
                           depth=args.depth, 
                           num_heads=args.num_heads,
                           decoder_embed_dim=args.decoder_embed_dim, 
                           decoder_depth=args.decoder_depth, 
                           decoder_num_heads=args.decoder_num_heads, 
                           mlp_ratio=args.mlp_ratio)

    elif args.model_name == "cross_vit":
        model = mae_vit_cross_vit(img_size=args.img_size, 
                           patch_size=args.patch_size, 
                           in_chans=args.in_chans, 
                           embed_dim=args.embed_dim, 
                           depth=args.depth, 
                           num_heads=args.num_heads,
                           decoder_embed_dim=args.decoder_embed_dim, 
                           decoder_depth=args.decoder_depth, 
                           decoder_num_heads=args.decoder_num_heads, 
                           mlp_ratio=args.mlp_ratio)
    return model
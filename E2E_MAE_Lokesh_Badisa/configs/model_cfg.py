

#ViT configs
small = {
'patch_size': 5,
'embed_dim': 384,
'encoder_depth': 12,
'encoder_heads': 6,
'decoder_embed_dim': 512,
'decoder_depth': 8,
'decoder_heads': 16,
'mlp_ratio': 4,
}

base = {
'patch_size': 5,
'embed_dim': 768,
'encoder_depth': 12,
'encoder_heads': 12,
'decoder_embed_dim': 512,
'decoder_depth': 8,
'decoder_heads': 16,
'mlp_ratio': 4,
}

large = {
'patch_size': 5,
'embed_dim': 1024,
'encoder_depth': 24,
'encoder_heads': 16,
'decoder_embed_dim': 512,
'decoder_depth': 8,
'decoder_heads': 16,
'mlp_ratio': 4,
}

huge = {
'patch_size': 5,
'embed_dim': 1280,
'encoder_depth': 32,
'encoder_heads': 16,
'decoder_embed_dim': 512,
'decoder_depth': 8,
'decoder_heads': 16,
'mlp_ratio': 4,
}
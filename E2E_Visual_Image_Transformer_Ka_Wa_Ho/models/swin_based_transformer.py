""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

S3 (AutoFormerV2, https://arxiv.org/abs/2111.14725) Swin weights from
    - https://github.com/microsoft/Cream/tree/main/AutoFormerV2

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman

Additional modifications are added by Ka Wa, Ho for GSoC 2023
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import logging
import math
from typing import Callable, List, Optional, Tuple, Union
from itertools import repeat

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, ClassifierHead, to_2tuple, to_ntuple, trunc_normal_, \
    _assert, use_fused_attn
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.vision_transformer import get_init_weights_vit

__all__ = ['SwinTransformer']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)  # NOTE not tested for prime-time yet

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: List[int] = [7],
            shift_size: List[int] = [0],
            shift_win: bool = True,
            conv_win: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_types = len(window_size)
        self.window_size = list(window_size)
        self.shift_size = shift_size
        self.shift_win = shift_win
        self.mlp_ratio = mlp_ratio
        for _ in range(self.window_types): 
            if min(self.input_resolution) <= self.window_size[_]:
                # if window size is larger than input resolution, we don't partition windows
                self.shift_size[_] = 0
                self.window_size[_] = min(self.input_resolution)
            assert 0 <= self.shift_size[_] < self.window_size[_], "shift_size must in 0-window_size"

            if (self.shift_size[_] > 0) and self.shift_win:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                cnt = 0
                for h in (
                        slice(0, -self.window_size[_]),
                        slice(-self.window_size[_], -self.shift_size[_]),
                        slice(-self.shift_size[_], None)):
                    for w in (
                            slice(0, -self.window_size[_]),
                            slice(-self.window_size[_], -self.shift_size[_]),
                            slice(-self.shift_size[_], None)):
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, self.window_size[_])  # num_win, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, self.window_size[_] * self.window_size[_])
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None
            self.register_buffer(f"attn_mask{_}", attn_mask)

        self.norm1 = norm_layer(dim)

        self.attn = torch.nn.ModuleList([WindowAttention(
            dim//self.window_types,
            num_heads=num_heads//self.window_types,
            head_dim=head_dim,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        ) for window_size in self.window_size])

        self.conv_win = conv_win 

        if self.conv_win:
            self.conv = torch.nn.ModuleList([torch.nn.Conv2d(
                in_channels=dim//self.window_types,
                out_channels=dim//self.window_types,
                kernel_size=window_size-1,
                groups=dim//self.window_types,
                padding='same'
            ) for window_size in self.window_size])
            self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        B, H, W, C = x.shape
        _assert(H == self.input_resolution[0], "input feature has wrong size")
        _assert(W == self.input_resolution[1], "input feature has wrong size")

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        shifted_x_list = []
        for _ in range(self.window_types): 
            if (self.shift_size[_] > 0) and self.shift_win:
                shifted_x =  torch.roll(x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types], shifts=(-self.shift_size[_], -self.shift_size[_]), dims=(1, 2))
            else:
                shifted_x = x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types]

            x_windows = window_partition(shifted_x, self.window_size[_])  # num_win*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size[_] * self.window_size[_], C//self.window_types)  # num_win*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn[_](x_windows, mask=getattr(self, f"attn_mask{_}"))  # num_win*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size[_], self.window_size[_], C//self.window_types)
            shifted_x = window_reverse(attn_windows, self.window_size[_], H, W)  # B H' W' C

            # reverse cyclic shift
            if (self.shift_size[_] > 0) and self.shift_win:
                shifted_x_list.append(torch.roll(shifted_x, shifts=(self.shift_size[_], self.shift_size[_]), dims=(1, 2)))
            else:
                shifted_x_list.append(shifted_x)

        x = torch.cat(shifted_x_list, 3)
        x = shortcut + self.drop_path(x)

        # depthwise conv
        if self.conv_win:
            x = self.norm2(x)
            for _ in range(self.window_types):
              x_sub = x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types] 
              shifted_x_list[_] = self.conv[_](x_sub.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + torch.cat(shifted_x_list, 3)

        # FFN
        x = x.reshape(B, -1, C)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            shift_win: bool = True,
            conv_win: bool = False,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: Union[List[int], int] = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        if not isinstance(window_size, list): window_size = [window_size]

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=[0 for _ in window_size] if (i % 2 == 0) else [_ // 2 for _ in window_size],
                shift_win=shift_win, 
                conv_win=conv_win, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            shift_win: bool = True,
            conv_win: bool = False,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size: Union[List[int], int] = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            weight_init: str = '',
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            head_dim: Dimension of self-attention heads.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )
        self.patch_grid = self.patch_embed.grid_size

        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = list(repeat(window_size, self.num_layers))
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_grid[0] // scale,
                    self.patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                shift_win=shift_win,
                conv_win=conv_win,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    if 'head.fc.weight' in state_dict:
        return state_dict
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        k = k.replace('head.', 'head.fc.')
        out_dict[k] = v
    return out_dict


def _create_swin_transformer(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }

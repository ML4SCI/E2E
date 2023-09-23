"""
Code copied and modified from 
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
"""

import collections.abc
from functools import partial
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from ..layers import StochasticDepth, WindowAttention
from . import utils
from .mlp import mlp_block


class SwinTransformerBlock(keras.Model):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (tuple): Window size.
        conv_win (bool): Apply conv as in Win models.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer.  Default: layers.LayerNormalization
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=4,
        head_dim=None,
        window_size=(7),
        conv_win=True,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = list(window_size)
        self.window_types = len(self.window_size)
        self.shift_size = list(shift_size)
        self.conv_win = conv_win
        self.mlp_ratio = mlp_ratio
        self.attn, self.conv = [], []
        for _ in range(self.window_types):
            if min(self.input_resolution) <= self.window_size[_]:
                # if window size is larger than input resolution, we don't partition windows
                self.shift_size[_] = 0
                self.window_size[_] = min(self.input_resolution)
            assert (
                0 <= self.shift_size[_] < self.window_size[_]
            ), "shift_size must in 0-window_size"
            self.attn.append(
                    WindowAttention(
                        dim//self.window_types,
                        num_heads=num_heads//self.window_types,
                        head_dim=head_dim,
                        window_size=self.window_size[_]
                        if isinstance(self.window_size[_], collections.abc.Iterable)
                        else (self.window_size[_], self.window_size[_]),
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        name=f"window_attention_{_}",
                    )
            )
            
            self.conv.append(
                    tf.keras.layers.DepthwiseConv2D(
                        kernel_size=self.window_size[_]+1,
                        padding='same'
                    )
            )


        self.norm1 = norm_layer()
        self.norm2 = norm_layer()
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else tf.identity
        )
        self.norm3 = norm_layer()
        self.mlp = mlp_block(
            dropout_rate=drop, hidden_units=[int(dim * mlp_ratio), dim]
        )

        # `get_attn_mask()` uses NumPy to make in-place assignments.
        # Since this is done during initialization, it's okay.
        self.attn_mask = self.get_attn_mask()

    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = [np.zeros((1, H, W, 1)) for _ in range(self.window_types)] # [1, H, W, 1]
        attn_mask_list = []
        for _ in range(self.window_types):
            if self.shift_size[_] > 0:
                cnt = 0
                for h in (
                    slice(0, -self.window_size[_]),
                    slice(-self.window_size[_], -self.shift_size[_]),
                    slice(-self.shift_size[_], None),
                ):
                    for w in (
                        slice(0, -self.window_size[_]),
                        slice(-self.window_size[_], -self.shift_size[_]),
                        slice(-self.shift_size[_], None),
                    ):
                        img_mask[_][:, h, w, :] = cnt
                        cnt += 1

                img_mask[_] = tf.convert_to_tensor(img_mask[_], dtype="float32")
                mask_windows = utils.window_partition(
                    img_mask[_], self.window_size[_]
                )  # [num_win, window_size, window_size, 1]
                mask_windows = tf.reshape(
                    mask_windows, (-1, self.window_size[_] * self.window_size[_])
                )
                attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
                    mask_windows, 2
                )
                attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
                attn_mask_list.append(tf.where(attn_mask == 0, 0.0, attn_mask))
            else:
                attn_mask_list.append(None)
        return attn_mask_list

    def call(
        self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        x_list = []
        for _ in range(self.window_types):
            # cyclic shift
            if self.shift_size[_] > 0:
                shifted_x = tf.roll(
                    x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types], shift=(-self.shift_size[_], -self.shift_size[_]), axis=(1, 2)
                )
            else:
                shifted_x = x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types]

            # partition windows
            x_windows = utils.window_partition(
                shifted_x, self.window_size[_]
            )  # [num_win*B, window_size, window_size, C]
            x_windows = tf.reshape(
                x_windows, (-1, self.window_size[_] * self.window_size[_], C//self.window_types)
            )  # [num_win*B, window_size*window_size, C]
    
            # W-MSA/SW-MSA
            if not return_attns:
                attn_windows = self.attn[_](
                    x_windows, mask=self.attn_mask[_]
                )  # [num_win*B, window_size*window_size, C]
            else:
                attn_windows, attn_scores = self.attn[_](
                    x_windows, mask=self.attn_mask[_], return_attns=True
                )  # [num_win*B, window_size*window_size, C]
            # merge windows
            attn_windows = tf.reshape(
                attn_windows, (-1, self.window_size[_], self.window_size[_], C//self.window_types)
            )
            shifted_x = utils.window_reverse(
                attn_windows, self.window_size[_], H, W
            )  # [B, H', W', C]
    
            # reverse cyclic shift
            if self.shift_size[_] > 0:
                x_sub = tf.roll(
                    shifted_x,
                    shift=(self.shift_size[_], self.shift_size[_]),
                    axis=(1, 2),
                )
            else:
                x_sub = shifted_x

            x_list.append(x_sub)

        x = tf.concat(x_list, 3)

        if self.conv_win:
            x = tf.reshape(shortcut, (B, H, W, C)) + self.drop_path(x)
            x = self.norm2(x)
            for _ in range(self.window_types):
                x_sub = x[:,:,:,_*C//self.window_types:(_+1)*C//self.window_types]
                x_list[_] = self.conv[_](x_sub)

            x = x + tf.concat(x_list, 3)

        # FFN
        x = tf.reshape(x, (B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        if return_attns:
            return x, attn_scores
        else:
            return x

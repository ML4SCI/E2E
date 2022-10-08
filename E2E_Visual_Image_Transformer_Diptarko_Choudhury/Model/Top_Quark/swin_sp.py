import sys
import os

# import Initializer
from . import Initializer 
import collections.abc
from functools import partial
from itertools import repeat
from typing import Dict, Union, List, Tuple
import numpy as np




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers as L


import collections.abc
from functools import partial
from itertools import repeat
from typing import Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L



# https://github.com/rwightman/pytorch-image-models/blob/6d4665bb52390974e0cf9674c60c41946d2f4ee2/timm/models/layers/helpers.py#L10

def to_ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    xx, yy = tf.meshgrid(range(win_h), range(win_w))
    coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
    coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # [2, Wh*Ww, Wh*Ww]
    relative_coords = tf.transpose(
        relative_coords, perm=[1, 2, 0]
    )  # [Wh*Ww, Wh*Ww, 2]

    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = tf.stack([xx, yy], axis=-1)

    return tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]
class WindowAttention(layers.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )  # Wh, Ww
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        self.relative_position_index = get_relative_position_index(
            self.win_h, self.win_w
        )

        self.qkv = layers.Dense(
            self.attn_dim * 3, use_bias=qkv_bias, name="attention_qkv",
             kernel_initializer= Initializer.MLP_Normal()
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name="attention_projection",
         kernel_initializer= Initializer.MLP_Normal())
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer=Initializer.Relative_pos_b_t(),
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> tf.Tensor:
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def call(
        self, x, mask=None, return_attns=False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tf.unstack(qkv, 3)

        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + tf.expand_dims(mask, 1)[None, ...]

            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, -1)
        else:
            attn = tf.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "win_h": self.win_h,
                "win_w": self.win_w,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "attn_dim": self.attn_dim,
                "scale": self.scale,
            }
        )
        return config
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prop)

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob

            # shape = (tf.shape(x)[0],) + (1,) * (tf.shape(tf.shape(x)) - 1)
            shape = (x.shape[0],)+(1,)*(tf.shape(x).shape[0]-1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config
class PatchMerging(L.Layer):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(
        self,
        input_resolution,
        dim,
        out_dim=None,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer()
        self.reduction = L.Dense(self.out_dim, use_bias=False, kernel_initializer=Initializer.MLP_Normal())

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        x = tf.reshape(x, (B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, (B, -1, 4 * C))  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "dim": self.dim,
                "out_dim": self.out_dim,
                "norm": self.norm,
            }
        )
        return config
def window_partition(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, (-1, window_size, window_size, C))
    return windows
def window_reverse(windows: tf.Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = tf.shape(windows)[0] // tf.cast(
        H * W / window_size / window_size, dtype="int32"
    )

    x = tf.reshape(
        windows,
        (
            B,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))
class SwinTransformerBlock(keras.Model):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
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
        window_size=7,
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
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="window_attention",
        )

        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else tf.identity
        )
        self.norm2 = norm_layer()
        self.mlp = mlp_block(
            dropout_rate=drop, hidden_units=[int(dim * mlp_ratio), dim]
        )

        if self.shift_size > 0:
            # `get_attn_mask()` uses NumPy to make in-place assignments.
            # Since this is done during initialization, it's okay.
            self.attn_mask = self.get_attn_mask()
        else:
            self.attn_mask = None

    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = np.zeros((1, H, W, 1))  # [1, H, W, 1]
        cnt = 0
        for h in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype="float32")
        mask_windows = window_partition(
            img_mask, self.window_size
        )  # [num_win, window_size, window_size, 1]
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
            mask_windows, 2
        )
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        return tf.where(attn_mask == 0, 0.0, attn_mask)

    def call(
        self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:


        H, W = self.input_resolution

        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        if not return_attns:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        else:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=self.attn_mask, return_attns=True
            )  # [num_win*B, window_size*window_size, C]
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attns:
            return x, attn_scores
        else:
            return x
class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer. Default: layers.LayerNormalization
        downsample (layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
        depth,
        num_heads=4,
        head_dim=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        downsample=None,
        **kwargs,
    ):

        super().__init__(kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        blocks = [
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                name=f"swin_transformer_block_{i}",
            )
            for i in range(depth)
        ]
        self.blocks = blocks

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

    def call(
        self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if return_attns:
            attention_scores = {}

        for i, block in enumerate(self.blocks):
            if not return_attns:
                x = block(x)

            else:
                x, attns = block(x, return_attns)
                attention_scores.update({f"swin_block_{i}": attns})
        if self.downsample is not None:
            x = self.downsample(x)

        if return_attns:
            return x, attention_scores
        else:
            return x
def mlp_block(dropout_rate: float, hidden_units: List[int], name: str = "mlp"):
    """FFN for a Transformer block."""
    ffn = keras.Sequential(name=name)
    for (idx, units) in enumerate(hidden_units):
        ffn.add(
            layers.Dense(
                units,
                activation=tf.nn.gelu if idx == 0 else None,
                kernel_initializer=Initializer.MLP_Normal(),
            )
        )
        ffn.add(layers.Dropout(dropout_rate))
    return ffn
class SwinTransformer(keras.Model):
    """Swin Transformer
        A TensorFlow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (layers.Layer): Normalization layer. Default: layers.LayerNormalization.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pre_logits (bool): If True, return model without classification head. Default: False
    """

    def __init__(
        self,
        img_size=224,
        part_model=False,
        eff_blocks=None,
        partition_=None,
        patch_size=4,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        head_dim=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        ape=False,
        patch_norm=True,
        pre_logits=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (part_model is False and eff_blocks is None and partition_ is None) or (part_model is True and eff_blocks is not None and partition_ is not None), "Contradictory nature of eff_blocks, part_model, and partition_"
        assert (partition_ == 1 or partition_ == 2 or partition_ is None), f"partition_ was given value {partition_} it only accepts 1, 2, None"



        self.img_size = (
            img_size
            if isinstance(img_size, collections.abc.Iterable)
            else (img_size, img_size)
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        self.eff_blocks = eff_blocks
        self.partition_ = partition_
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.ape = ape

        # split image into non-overlapping patches


        self.patch_grid = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = tf.Variable(
                tf.zeros((1, self.num_patches, self.embed_dim)),
                trainable=True,
                name="absolute_pos_embed",
            )
        else:
            self.absolute_pos_embed = None
        self.pos_drop = L.Dropout(drop_rate)

        # build layers
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = [
                int(self.embed_dim * 2 ** i) for i in range(self.num_layers)
            ]
        embed_out_dim = self.embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [
            float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule


        if part_model:
            if self.partition_ == 1:
                layers = [
                    BasicLayer(
                        dim=self.embed_dim[i],
                        out_dim=embed_out_dim[i],
                        input_resolution=(
                            self.patch_grid[0] // (2 ** i),
                            self.patch_grid[1] // (2 ** i),
                        ),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        head_dim=head_dim[i],
                        window_size=window_size[i],
                        mlp_ratio=mlp_ratio[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i < self.num_layers - 1) else None,
                        name=f"basic_layer_{i}",
                    )
                    for i in range(self.eff_blocks)
                ]

                self.projection = keras.Sequential(
                    [
                        L.Conv2D(
                            filters=embed_dim,
                            kernel_size=(patch_size, patch_size),
                            strides=(patch_size, patch_size),
                            padding="VALID",
                            name="conv_projection",
                            kernel_initializer=Initializer.Kaimming_Uniform(),
                            bias_initializer=Initializer.Kaimming_Uniform()
                        ),
                        L.Reshape(
                            target_shape=(-1, embed_dim),
                            name="flatten_projection",
                        ),
                    ],
                    name="projection",
                )
                if patch_norm:
                    self.projection.add(norm_layer())

            if self.partition_ == 2:
                layers = [
                    BasicLayer(
                        dim=self.embed_dim[i],
                        out_dim=embed_out_dim[i],
                        input_resolution=(
                            self.patch_grid[0] // (2 ** i),
                            self.patch_grid[1] // (2 ** i),),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        head_dim=head_dim[i],
                        window_size=window_size[i],
                        mlp_ratio=mlp_ratio[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i < self.num_layers - 1) else None,
                        name=f"basic_layer_{i}",
                        )
                        for i in range(self.eff_blocks, self.num_layers,1)
                        ]

        else:
            layers = [
                    BasicLayer(
                        dim=self.embed_dim[i],
                        out_dim=embed_out_dim[i],
                        input_resolution=(
                            self.patch_grid[0] // (2 ** i),
                            self.patch_grid[1] // (2 ** i),
                        ),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        head_dim=head_dim[i],
                        window_size=window_size[i],
                        mlp_ratio=mlp_ratio[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i < self.num_layers - 1) else None,
                        name=f"basic_layer_{i}",
                    )
                        for i in range(self.num_layers)
                ]

            self.projection = keras.Sequential(
                [
                    L.Conv2D(
                        filters=embed_dim,
                        kernel_size=(patch_size, patch_size),
                        strides=(patch_size, patch_size),
                        padding="VALID",
                        name="conv_projection",
                        kernel_initializer=Initializer.Kaimming_Uniform(),
                        bias_initializer=Initializer.Kaimming_Uniform(),
                    ),
                    L.Reshape(
                        target_shape=(-1, embed_dim),
                        name="flatten_projection",
                    ),
                ],
                name="projection",
            )
            if patch_norm:
                self.projection.add(norm_layer())

        self.swin_layers = layers

        self.norm = norm_layer()

        self.pre_logits = pre_logits
        if not self.pre_logits:
            self.head = L.Dense(num_classes, name="classification_head")

    def forward_features(self, x):
        if self.eff_blocks is None:
            x = self.projection(x)
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            for swin_layer in self.swin_layers:
                x = swin_layer(x)

            x = self.norm(x)  # [B, L, C]
            return x

        else:
            if self.partition_ == 1:
                x = self.projection(x)
                if self.absolute_pos_embed is not None:
                    x = x + self.absolute_pos_embed
                x = self.pos_drop(x)
                for swin_layer in self.swin_layers:
                    x = swin_layer(x)
                return x
            if self.partition_ == 2:
                for swin_layer in self.swin_layers:
                    x = swin_layer(x)
                x = self.norm(x)
                return x


    def forward_head(self, x):

        x = tf.reduce_mean(x, axis=1)
        return x if self.pre_logits else self.head(x)

    def call(self, x):
        x = self.forward_features(x)
        if self.partition_ == 2:
            x = self.forward_head(x)
        if self.partition_ == None:
            x = self.forward_head(x)
        return x

    # Thanks to Willi Gierke for this suggestion.
    @tf.function(
        input_signature=[tf.TensorSpec([None, None, None, 8], tf.float32)]
    )
    def get_attention_scores(
        self, x: tf.Tensor
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        all_attention_scores = {}

        x = self.projection(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, swin_layer in enumerate(self.swin_layers):
            x, attention_scores = swin_layer(x, return_attns=True)
            all_attention_scores.update({f"swin_stage_{i}": attention_scores})

        return all_attention_scores

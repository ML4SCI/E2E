import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Conv2dNormActivation, MLP
# from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


class CrissCrossMultiheadSelfAttention(nn.Module):
    def __init__(self, k_factor, num_heads, in_channels):
        super(CrissCrossMultiheadSelfAttention, self).__init__()
        self.k_factor = k_factor
        self.num_heads = num_heads
        self.in_channels = in_channels


        self.qkv_linear = nn.ModuleList()
        for _ in range(in_channels):
            self.qkv_linear.append(nn.Linear(k_factor, k_factor * 3))
        self.fc_out = nn.Linear(k_factor, k_factor)

    def forward(self, x):
        '''
        The following input shape can be achieved by modifying the convolutional layers
        in the ViT preprocessing to be depth-wise with in_channels = K*in_channels and
        groups = in_channels where K is the scale factor of how many convolutional
        filters you want for each image channel
        V,K,Q input shape: batch_size, length, hidden_dim*channels
        '''
        N = x.shape[0]

        #channels shape: [[batch_size, length, hidden_dim], ...] list length=in_channels
        in_channels = x.chunk(self.in_channels, dim=2)

        queries = []
        keys = []
        values = []
        for channel, qkv in zip(in_channels, self.qkv_linear):
            q, k, v = qkv(channel).chunk(3, dim=-1)
            queries.append(q)
            keys.append(k)
            values.append(v)

        #q, k before: [[batch_size, length, hidden_dim], ...] list length=in_channels
        queries = torch.stack(queries, -1).unsqueeze(-1)
        keys = torch.stack(keys, -1).unsqueeze(-2)
        values = torch.stack(values, -1)
        #q shape after: (batch_size, length, hidden_dim, in_channels, 1)
        #k shape after: (batch_size, length, hidden_dim, 1, in_channels)

        #scores shape: (batch_size, length, hidden_dim, in_channels, in_channels)
        scores = torch.matmul(queries, keys) / (self.k_factor ** 0.5)
        attention = F.softmax(scores, -1)

        #context shape: (batch_size, length, hidden_dim, in_channels)
        context = torch.einsum('ijklm,ijkl->ijkm', attention, values)

        #context shape: (batch_size, length, in_channels, hidden_dim)
        context = context.transpose(-2, -1)

        out = self.fc_out(context)

        #out shape: (batch_size, length, in_channels * hidden_dim)
        out = out.flatten(start_dim=2)

        return out

class CrissCrossMultiheadCrossAttention(nn.Module):
    def __init__(self, k_factor, num_heads, in_channels):
        super(CrissCrossMultiheadCrossAttention, self).__init__()
        self.k_factor = k_factor
        self.num_heads = num_heads
        self.in_channels = in_channels


        self.q_linear = nn.ModuleList()
        self.kv_linear = nn.ModuleList()
        for _ in range(in_channels):
            self.q_linear.append(nn.Linear(k_factor, k_factor))
            self.kv_linear.append(nn.Linear(k_factor, k_factor * 2))
        self.fc_out = nn.Linear(k_factor, k_factor)

    def forward(self, x, context):
        '''
        The following input shape can be achieved by modifying the convolutional layers
        in the ViT preprocessing to be depth-wise with in_channels = K*in_channels and
        groups = in_channels where K is the scale factor of how many convolutional
        filters you want for each image channel
        V,K,Q input shape: batch_size, length, hidden_dim*channels
        '''
        N = x.shape[0]

        #channels shape: [[batch_size, length, hidden_dim], ...] list length=in_channels
        in_channels = x.chunk(self.in_channels, dim=2)

        context_in_channels = context.chunk(self.in_channels, dim=2)

        queries = []
        keys = []
        values = []
        for channel, cont_channel, q_lin, kv_lin in zip(in_channels, context_in_channels, self.q_linear, self.kv_linear):
            q = q_lin(channel)
            k, v = kv_lin(cont_channel).chunk(2, dim=-1)
            queries.append(q)
            keys.append(k)
            values.append(v)

        #q, k before: [[batch_size, length, hidden_dim], ...] list length=in_channels
        queries = torch.stack(queries, -1).unsqueeze(-1)
        keys = torch.stack(keys, -1).unsqueeze(-2)
        values = torch.stack(values, -1)
        #q shape after: (batch_size, length, hidden_dim, in_channels, 1)
        #k shape after: (batch_size, length, hidden_dim, 1, in_channels)

        #scores shape: (batch_size, length, hidden_dim, in_channels, in_channels)
        scores = torch.matmul(queries, keys) / (self.k_factor ** 0.5)
        attention = F.softmax(scores, -1)

        #context shape: (batch_size, length, hidden_dim, in_channels)
        context = torch.einsum('ijklm,ijkl->ijkm', attention, values)

        #context shape: (batch_size, length, in_channels, hidden_dim)
        context = context.transpose(-2, -1)

        out = self.fc_out(context)

        #out shape: (batch_size, length, in_channels * hidden_dim)
        out = out.flatten(start_dim=2)

        return out


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        in_channels: int,
        k_factor: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = CrissCrossMultiheadSelfAttention(k_factor, num_heads, in_channels)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_channels: int,
        k_factor: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.enc_pos_embedding = nn.Parameter(torch.empty(1, seq_length + 1, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                num_channels,
                k_factor,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, input: torch.Tensor, mask_ratio=0.5):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.enc_pos_embedding[:, 1:, :]
        input, mask, ids_restore = self.random_masking(input, mask_ratio)
        cls_token = self.cls_token + self.enc_pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(input.shape[0], -1, -1)
        input = torch.cat((cls_tokens, input), dim=1)
        return self.ln(self.layers(self.dropout(input))), mask, ids_restore


class DecoderBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        in_channels: int,
        k_factor: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = CrissCrossMultiheadSelfAttention(k_factor, num_heads, in_channels)
        self.ln_2 = norm_layer(hidden_dim)
        self.cross_attention = CrissCrossMultiheadCrossAttention(k_factor, num_heads, in_channels)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_3 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.ln_1(input)
        x = self.self_attention(input)
        input = input + self.dropout(x)
        x = self.ln_2(input)
        x = self.cross_attention(x, context)
        input = input + self.dropout(x)

        y = self.ln_3(x)
        y = self.mlp(y)
        return x + y


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_channels: int,
        k_factor: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dec_pos_embedding = nn.Parameter(torch.empty(1, seq_length + 1, hidden_dim).normal_(std=0.02))  # from BERT
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = DecoderBlock(
                num_heads,
                hidden_dim,
                num_channels,
                k_factor,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor, ids_restore: torch.Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        cls_token = self.cls_token + self.dec_pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        mask_tokens = self.mask_token.repeat(context.shape[0], ids_restore.shape[1] + 1 - context.shape[1], 1)
        context_ = torch.cat([context[:, 1:, :], mask_tokens], dim=1)  # no cls token
        context_ = torch.gather(context_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context.shape[2]))  # unshuffle
        context = torch.cat([context[:, :1, :], context_], dim=1)  # append cls token
        context = context + self.dec_pos_embedding

        for l in self.layers:
            x = l(x, context)
        return self.ln(x)



class DepthwiseCrossViTMAE(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        k_factor: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.k_factor = k_factor
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.mask_ratio = mask_ratio

        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, groups=in_channels, out_channels=in_channels*k_factor, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = Encoder(
            seq_length,
            num_layers,
            in_channels,
            k_factor,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        self.decoder = Decoder(
            seq_length,
            num_layers,
            in_channels,
            k_factor,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, in_channels * k_factor, n_h, n_w) -> (n, in_channels * k_factor, (n_h * n_w))
        x = x.reshape(n, self.in_channels * self.k_factor, n_h * n_w)

        # (n, in_channels * k_factor, (n_h * n_w)) -> (n, (n_h * n_w), in_channels * k_factor)
        # The self attention layer expects inputs in the format (N, L, D)
        # where L is the source sequence length, N is the batch size, D is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch


        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs: torch.Tensor):
        # Reshape and permute the input tensor
        imgs = self._process_input(imgs)
        n = imgs.shape[0]

        enc_output, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        x = self.decoder(imgs, enc_output, ids_restore)

        # Remove the class token
        x = x[:, 1:, :]

        loss = self.forward_loss(imgs, x, mask)
        return loss, x, mask


def _vision_transformer(
    patch_size: int,
    in_channels: int,
    k_factor: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DepthwiseCrossViTMAE:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 125)

    model = DepthwiseCrossViTMAE(
        image_size=image_size,
        in_channels=in_channels,
        k_factor=k_factor,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

model = _vision_transformer(
                            patch_size=5,
                            in_channels=8,
                            k_factor=32,
                            num_layers=4,
                            num_heads=4,
                            hidden_dim=256,
                            mlp_dim=256,
                            weights=None,
                            progress=False,
                            )
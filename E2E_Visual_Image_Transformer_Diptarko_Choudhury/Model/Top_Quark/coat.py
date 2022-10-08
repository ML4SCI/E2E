import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import os
import sys

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
TF_BATCH_NORM_EPSILON = 0.001
LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def reload_model_weights(model, pretrained_dict, sub_release, pretrained="imagenet", mismatch_class=None, request_resolution=-1, method="nearest"):
    if pretrained is None:
        return
    if isinstance(pretrained, str) and pretrained.endswith(".h5"):
        print(">>>> Load pretrained from:", pretrained)
        # model.load_weights(pretrained, by_name=True, skip_mismatch=True)
        load_weights_with_mismatch(model, pretrained, mismatch_class, request_resolution, method)
        return pretrained

    file_hash = pretrained_dict.get(model.name, {}).get(pretrained, None)
    if file_hash is None:
        print(">>>> No pretrained available, model will be randomly initialized")
        return None

    if isinstance(file_hash, dict):
        # file_hash is a dict like {224: "aa", 384: "bb", 480: "cc"}
        if request_resolution == -1:
            input_height = model.input_shape[1]
            if input_height is None:  # input_shape is (None, None, 3)
                request_resolution = max(file_hash.keys())
            else:
                request_resolution = min(file_hash.keys(), key=lambda ii: abs(ii - input_height))
        pretrained = "{}_".format(request_resolution) + pretrained
        file_hash = file_hash[request_resolution]
        # print(f"{request_resolution = }, {pretrained = }, {file_hash = }")
    elif request_resolution == -1:
        request_resolution = 224  # Default is 224

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.h5"
    url = pre_url.format(sub_release, model.name, pretrained)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return None
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        # model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        load_weights_with_mismatch(model, pretrained_model, mismatch_class, request_resolution, method)
        return pretrained_model



""" Wrapper for default parameters """


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_swish(inputs):
    """`out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244"""
    return inputs * tf.nn.relu6(inputs + 3) / 6


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_sigmoid_torch(inputs):
    """https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    toch.nn.Hardsigmoid: 0 if x <= −3 else (1 if x >= 3 else x / 6 + 1/2)
    keras.activations.hard_sigmoid: 0 if x <= −2.5 else (1 if x >= 2.5 else x / 5 + 1/2) -> tf.clip_by_value(inputs / 5 + 0.5, 0, 1)
    """
    return tf.clip_by_value(inputs / 6 + 0.5, 0, 1)


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def mish(inputs):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    Copied from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/mish.py
    """
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def phish(inputs):
    """Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    Paper: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
    return inputs * tf.math.tanh(tf.nn.gelu(inputs))


def activation_by_name(inputs, activation="relu", name=None):
    """Typical Activation layer added hard_swish and prelu."""
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation == "mish":
        return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation == "phish":
        return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation.lower().startswith("gelu/app"):
        # gelu/approximate
        return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation.lower() == ("hard_sigmoid_torch"):
        return keras.layers.Activation(activation=hard_sigmoid_torch, name=layer_name)(inputs)
    else:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
class EvoNormalization(tf.keras.layers.Layer):
    def __init__(self, nonlinearity=True, num_groups=-1, zero_gamma=False, momentum=0.99, epsilon=0.001, data_format="auto", **kwargs):
        # [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py)
        # EVONORM_B0: nonlinearity=True, num_groups=-1
        # EVONORM_S0: nonlinearity=True, num_groups > 0
        # EVONORM_B0 / EVONORM_S0 linearity: nonlinearity=False, num_groups=-1
        # EVONORM_S0A linearity: nonlinearity=False, num_groups > 0
        super().__init__(**kwargs)
        self.data_format, self.nonlinearity, self.zero_gamma, self.num_groups = data_format, nonlinearity, zero_gamma, num_groups
        self.momentum, self.epsilon = momentum, epsilon
        self.is_channels_first = True if data_format == "channels_first" or (data_format == "auto" and K.image_data_format() == "channels_first") else False

    def build(self, input_shape):
        all_axes = list(range(len(input_shape)))
        param_shape = [1] * len(input_shape)
        if self.is_channels_first:
            param_shape[1] = input_shape[1]
            self.reduction_axes = all_axes[:1] + all_axes[2:]
        else:
            param_shape[-1] = input_shape[-1]
            self.reduction_axes = all_axes[:-1]

        self.gamma = self.add_weight(name="gamma", shape=param_shape, initializer="zeros" if self.zero_gamma else "ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=param_shape, initializer="zeros", trainable=True)
        if self.num_groups <= 0:  # EVONORM_B0
            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                initializer="ones",
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )
        if self.nonlinearity:
            self.vv = self.add_weight(name="vv", shape=param_shape, initializer="ones", trainable=True)

        if self.num_groups > 0:  # EVONORM_S0
            channels_dim = input_shape[1] if self.is_channels_first else input_shape[-1]
            num_groups = int(self.num_groups)
            while num_groups > 1:
                if channels_dim % num_groups == 0:
                    break
                num_groups -= 1
            self.__num_groups__ = num_groups
            self.groups_dim = channels_dim // self.__num_groups__

            if self.is_channels_first:
                self.group_shape = [-1, self.__num_groups__, self.groups_dim, *input_shape[2:]]
                self.group_reduction_axes = list(range(2, len(self.group_shape)))  # [2, 3, 4]
                self.group_axes = 2
                self.var_shape = [-1, *param_shape[1:]]
            else:
                self.group_shape = [-1, *input_shape[1:-1], self.__num_groups__, self.groups_dim]
                self.group_reduction_axes = list(range(1, len(self.group_shape) - 2)) + [len(self.group_shape) - 1]  # [1, 2, 4]
                self.group_axes = -1
                self.var_shape = [-1, *param_shape[1:]]

    def __group_std__(self, inputs):
        # _group_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L171
        grouped = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped, self.group_reduction_axes, keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        std = tf.repeat(std, self.groups_dim, axis=self.group_axes)
        return tf.reshape(std, self.var_shape)

    def __batch_std__(self, inputs, training=None):
        # _batch_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L120
        def _call_train_():
            _, var = tf.nn.moments(inputs, self.reduction_axes, keepdims=True)
            # update_op = tf.assign_sub(moving_variance, (moving_variance - variance) * (1 - decay))
            delta = (self.moving_variance - var) * (1 - self.momentum)
            self.moving_variance.assign_sub(delta)
            return var

        def _call_test_():
            return self.moving_variance

        var = K.in_train_phase(_call_train_, _call_test_, training=training)
        return tf.sqrt(var + self.epsilon)

    def __instance_std__(self, inputs):
        # _instance_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L111
        # axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        _, var = tf.nn.moments(inputs, self.reduction_axes[1:], keepdims=True)
        return tf.sqrt(var + self.epsilon)

    def call(self, inputs, training=None, **kwargs):
        if self.nonlinearity and self.num_groups > 0:  # EVONORM_S0
            den = self.__group_std__(inputs)
            inputs = inputs * tf.nn.sigmoid(self.vv * inputs) / den
        elif self.num_groups > 0:  # EVONORM_S0a
            # EvoNorm2dS0a https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/evo_norm.py#L239
            den = self.__group_std__(inputs)
            inputs = inputs / den
        elif self.nonlinearity:  # EVONORM_B0
            left = self.__batch_std__(inputs, training)
            right = self.vv * inputs + self.__instance_std__(inputs)
            inputs = inputs / tf.maximum(left, right)
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nonlinearity": self.nonlinearity,
                "zero_gamma": self.zero_gamma,
                "num_groups": self.num_groups,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "data_format": self.data_format,
            }
        )
        return config


def batchnorm_with_activation(
    inputs, activation=None, zero_gamma=False, epsilon=1e-5, momentum=0.9, act_first=False, use_evo_norm=False, evo_norm_group_size=-1, name=None
):
    """Performs a batch normalization followed by an activation."""
    if use_evo_norm:
        nonlinearity = False if activation is None else True
        num_groups = inputs.shape[-1] // evo_norm_group_size  # Currently using gorup_size as parameter only
        return EvoNormalization(nonlinearity, num_groups=num_groups, zero_gamma=zero_gamma, epsilon=epsilon, momentum=momentum, name=name + "evo_norm")(inputs)

    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=momentum,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def layer_norm(inputs, zero_gamma=False, epsilon=LAYER_NORM_EPSILON, name=None):
    """Typical LayerNormalization with epsilon=1e-5"""
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, gamma_initializer=gamma_initializer, name=name and name + "ln")(inputs)


def group_norm(inputs, groups=32, epsilon=BATCH_NORM_EPSILON, name=None):
    """Typical GroupNormalization with epsilon=1e-5"""
    from tensorflow_addons.layers import GroupNormalization

    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return GroupNormalization(groups=groups, axis=norm_axis, epsilon=epsilon, name=name and name + "group_norm")(inputs)


def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """Typical Conv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    groups = max(1, groups)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "dw_pad")(inputs)
        padding = "VALID"
    return keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)


""" Blocks """


def output_block(inputs, filters=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True, act_first=False):
    nn = inputs
    if filters > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, filters, 1, strides=1, use_bias=act_first, use_torch_padding=is_torch_mode, name="features_")  # Also use_bias for act_first
        nn = batchnorm_with_activation(nn, activation=activation, act_first=act_first, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn


def global_context_module(inputs, use_attn=True, ratio=0.25, divisor=1, activation="relu", use_bias=True, name=None):
    """Global Context Attention Block, arxiv: https://arxiv.org/pdf/1904.11492.pdf"""
    height, width, filters = inputs.shape[1], inputs.shape[2], inputs.shape[-1]

    # activation could be ("relu", "hard_sigmoid")
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    reduction = make_divisible(filters * ratio, divisor, limit_round_down=0.0)

    if use_attn:
        attn = keras.layers.Conv2D(1, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "attn_conv")(inputs)
        attn = tf.reshape(attn, [-1, 1, 1, height * width])  # [batch, height, width, 1] -> [batch, 1, 1, height * width]
        attn = tf.nn.softmax(attn, axis=-1)
        context = tf.reshape(inputs, [-1, 1, height * width, filters])
        context = attn @ context  # [batch, 1, 1, filters]
    else:
        context = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    mlp = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_1_conv")(context)
    mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(mlp)
    mlp = activation_by_name(mlp, activation=hidden_activation, name=name)
    mlp = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_2_conv")(mlp)
    mlp = activation_by_name(mlp, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, mlp])


def se_module(inputs, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, use_conv=True, name=None):
    """Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf"""
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    # activation could be ("relu", "hard_sigmoid") for mobilenetv3
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    filters = inputs.shape[channel_axis]
    reduction = make_divisible(filters * se_ratio, divisor, limit_round_down=limit_round_down)
    # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    if use_conv:
        se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    else:
        se = keras.layers.Dense(reduction, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_dense")(se)
    se = activation_by_name(se, activation=hidden_activation, name=name)
    if use_conv:
        se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    else:
        se = keras.layers.Dense(filters, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_dense")(se)
    se = activation_by_name(se, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])


def eca_module(inputs, gamma=2.0, beta=1.0, name=None, **kwargs):
    """Efficient Channel Attention block, arxiv: https://arxiv.org/pdf/1910.03151.pdf"""
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    beta, gamma = float(beta), float(gamma)
    tt = int((tf.math.log(float(filters)) / tf.math.log(2.0) + beta) / gamma)
    kernel_size = max(tt if tt % 2 else tt + 1, 3)
    pad = kernel_size // 2

    nn = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=False)
    nn = tf.pad(nn, [[0, 0], [pad, pad]])
    nn = tf.expand_dims(nn, channel_axis)

    nn = keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="VALID", use_bias=False, name=name and name + "conv1d")(nn)
    nn = tf.squeeze(nn, axis=channel_axis)
    nn = activation_by_name(nn, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, nn])


def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """split drop connect rate in range `(start, end)` according to `num_blocks`"""
    drop_connect_rates = tf.split(tf.linspace(start, end, sum(num_blocks)), num_blocks)
    return [ii.numpy().tolist() for ii in drop_connect_rates]


def drop_block(inputs, drop_rate=0, name=None):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs


""" Other layers / functions """


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __anti_alias_downsample_initializer__(weight_shape, dtype="float32"):
    import numpy as np

    kernel_size, channel = weight_shape[0], weight_shape[2]
    ww = tf.cast(np.poly1d((0.5, 0.5)) ** (kernel_size - 1), dtype)
    ww = tf.expand_dims(ww, 0) * tf.expand_dims(ww, 1)
    ww = tf.repeat(ww[:, :, tf.newaxis, tf.newaxis], channel, axis=-2)
    return ww


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="SAME", trainable=False, name=None):
    """DepthwiseConv2D performing anti-aliasing downsample, arxiv: https://arxiv.org/pdf/1904.11486.pdf"""
    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        use_bias=False,
        trainable=trainable,
        depthwise_initializer=__anti_alias_downsample_initializer__,
        name=name and name + "anti_alias_down",
    )(inputs)


def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
    """Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < limit_round_down * vv:
        new_v += divisor
    return new_v


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __unfold_filters_initializer__(weight_shape, dtype="float32"):
    kernel_size = weight_shape[0]
    kernel_out = kernel_size * kernel_size
    ww = tf.reshape(tf.eye(kernel_out), [kernel_size, kernel_size, 1, kernel_out])
    if len(weight_shape) == 5:  # Conv3D or Conv3DTranspose
        ww = tf.expand_dims(ww, 2)
    return ww


def fold_by_conv2d_transpose(patches, output_shape=None, kernel_size=3, strides=2, dilation_rate=1, padding="SAME", compressed="auto", name=None):
    paded = kernel_size // 2 if padding else 0
    if compressed == "auto":
        compressed = True if len(patches.shape) == 4 else False

    if compressed:
        _, hh, ww, cc = patches.shape
        channel = cc // kernel_size // kernel_size
        conv_rr = tf.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    else:
        _, hh, ww, _, _, channel = patches.shape
        # conv_rr = patches
        conv_rr = tf.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    conv_rr = tf.transpose(conv_rr, [0, 3, 1, 2])  # [batch, channnel, hh * ww, kernel * kernel]
    conv_rr = tf.reshape(conv_rr, [-1, hh, ww, kernel_size * kernel_size])

    convtrans_rr = keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="VALID",
        output_padding=paded,
        use_bias=False,
        trainable=False,
        kernel_initializer=__unfold_filters_initializer__,
        name=name and name + "fold_convtrans",
    )(conv_rr)

    out = tf.reshape(convtrans_rr[..., 0], [-1, channel, convtrans_rr.shape[1], convtrans_rr.shape[2]])
    out = tf.transpose(out, [0, 2, 3, 1])
    if output_shape is None:
        output_shape = [-paded, -paded]
    else:
        output_shape = [output_shape[0] + paded, output_shape[1] + paded]
    out = out[:, paded : output_shape[0], paded : output_shape[1]]
    return out


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
class CompatibleExtractPatches(keras.layers.Layer):
    def __init__(self, sizes=3, strides=2, rates=1, padding="SAME", compressed=True, force_conv=False, **kwargs):
        super().__init__(**kwargs)
        self.sizes, self.strides, self.rates, self.padding = sizes, strides, rates, padding
        self.compressed, self.force_conv = compressed, force_conv

        self.kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
        self.strides = strides[1] if isinstance(strides, (list, tuple)) else strides
        self.dilation_rate = rates[1] if isinstance(rates, (list, tuple)) else rates
        self.filters = self.kernel_size * self.kernel_size

        if len(tf.config.experimental.list_logical_devices("TPU")) != 0 or self.force_conv:
            self.use_conv = True
        else:
            self.use_conv = False

    def build(self, input_shape):
        _, self.height, self.width, self.channel = input_shape
        if self.padding.upper() == "SAME":
            pad = self.kernel_size // 2
            self.pad_value = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
            self.height, self.width = self.height + pad * 2, self.width + pad * 2

        if self.use_conv:
            self.conv = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding="VALID",
                use_bias=False,
                trainable=False,
                kernel_initializer=__unfold_filters_initializer__,
                name=self.name and self.name + "unfold_conv",
            )
            self.conv.build([None, *input_shape[1:-1], 1])
        else:
            self._sizes_ = [1, self.kernel_size, self.kernel_size, 1]
            self._strides_ = [1, self.strides, self.strides, 1]
            self._rates_ = [1, self.dilation_rate, self.dilation_rate, 1]

    def call(self, inputs):
        if self.padding.upper() == "SAME":
            inputs = tf.pad(inputs, self.pad_value)

        if self.use_conv:
            merge_channel = tf.transpose(inputs, [0, 3, 1, 2])
            merge_channel = tf.reshape(merge_channel, [-1, self.height, self.width, 1])
            conv_rr = self.conv(merge_channel)

            # TFLite not supporting `tf.transpose` with len(perm) > 4...
            out = tf.reshape(conv_rr, [-1, self.channel, conv_rr.shape[1] * conv_rr.shape[2], self.filters])
            out = tf.transpose(out, [0, 2, 3, 1])  # [batch, hh * ww, kernel * kernel, channnel]
            if self.compressed:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.filters * self.channel])
            else:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.kernel_size, self.kernel_size, self.channel])
        else:
            out = tf.image.extract_patches(inputs, self._sizes_, self._strides_, self._rates_, "VALID")
            if not self.compressed:
                # [batch, hh, ww, kernel, kernel, channnel]
                out = tf.reshape(out, [-1, out.shape[1], out.shape[2], self.kernel_size, self.kernel_size, self.channel])
        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "sizes": self.sizes,
                "strides": self.strides,
                "rates": self.rates,
                "padding": self.padding,
                "compressed": self.compressed,
                "force_conv": self.force_conv,
            }
        )
        return base_config


class PreprocessInput:
    """`rescale_mode` `torch` means `(image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`, `tf` means `(image - 0.5) / 0.5`"""

    def __init__(self, input_shape=(224, 224, 3), rescale_mode="torch"):
        self.rescale_mode = rescale_mode
        self.input_shape = input_shape[1:-1] if len(input_shape) == 4 else input_shape[:2]

    def __call__(self, image, resize_method="bilinear", resize_antialias=False, input_shape=None):
        input_shape = self.input_shape if input_shape is None else input_shape[:2]
        image = tf.convert_to_tensor(image)
        if tf.reduce_max(image) < 2:
            image *= 255
        image = tf.image.resize(image, input_shape, method=resize_method, antialias=resize_antialias)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)

        if self.rescale_mode == "raw":
            return image
        elif self.rescale_mode == "raw01":
            return image / 255.0
        else:
            return tf.keras.applications.imagenet_utils.preprocess_input(image, mode=self.rescale_mode)


def imagenet_decode_predictions(preds, top=5):
    preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds
    return tf.keras.applications.imagenet_utils.decode_predictions(preds, top=top)


def add_pre_post_process(model, rescale_mode="tf", input_shape=None, post_process=None):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    model.preprocess_input = PreprocessInput(input_shape, rescale_mode=rescale_mode)
    model.decode_predictions = imagenet_decode_predictions if post_process is None else post_process
    model.rescale_mode = rescale_mode


#----------------------------------------------------------------------------
PRETRAINED_DICT = {"coatnet0": {"imagenet": {160: "bc4375d2f03b99ac4252770331f0d22f", 224: "29213248739d600cc526c11a79d06775"}}}


def mhsa_with_multi_head_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, global_query=None, out_shape=None, out_weight=True, qkv_bias=False, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    # vv_dim = out_shape // num_heads
    vv_dim = key_dim

    if global_query is not None:
        # kv = keras.layers.Dense(qk_out * 2, use_bias=qkv_bias, name=name and name + "kv")(inputs)
        kv = conv2d_no_bias(inputs, qk_out * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "kv_")
        kv = tf.reshape(kv, [-1, kv.shape[1] * kv.shape[2], kv.shape[-1]])
        key, value = tf.split(kv, [qk_out, out_shape], axis=-1)
        query = global_query
    else:
        # qkv = keras.layers.Dense(qk_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
        # qkv = conv2d_no_bias(inputs, qk_out * 2 + out_shape, kernel_size=1, name=name and name + "qkv_")
        qkv = conv2d_no_bias(inputs, qk_out * 3, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
        qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], qkv.shape[-1]])
        query, key, value = tf.split(qkv, [qk_out, qk_out, qk_out], axis=-1)
        # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }, {key_dim = }, {vv_dim = }")
        query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, hh * ww, hh * ww]
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {attention_scores.shape = }, {hh = }")
    attention_scores = MultiHeadRelativePositionalEmbedding(with_cls_token=False, attn_height=hh, name=name and name + "pos_emb")(attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, hh * ww, vv_dim], attention_output = [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    # attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(attention_output) if output_dropout > 0 else attention_output
    return attention_output


def res_MBConv(
    inputs,
    output_channel,
    conv_short_cut=True,
    strides=1,
    expansion=4,
    se_ratio=0,
    drop_rate=0,
    use_dw_strides=True,
    bn_act_first=False,
    activation="gelu",
    name="",
):
    """x ← Proj(Pool(x)) + Conv (DepthConv (Conv (Norm(x), stride = 2))))"""
    preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")

    if conv_short_cut:
        shortcut = keras.layers.MaxPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
        # shortcut = batchnorm_with_activation(shortcut, activation=activation, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    # MBConv
    input_channel = inputs.shape[-1]
    conv_strides, dw_strides = (1, strides) if use_dw_strides else (strides, 1)  # May swap stirdes with DW
    nn = conv2d_no_bias(preact, input_channel * expansion, 1, strides=conv_strides, use_bias=bn_act_first, padding="same", name=name + "expand_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(nn, 3, strides=dw_strides, use_bias=bn_act_first, padding="same", name=name + "MB_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, zero_gamma=False, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, activation=activation, name=name + "se_")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="same", name=name + "MB_pw_")
    # nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "MB_pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return keras.layers.Add(name=name + "output")([shortcut, nn])


def res_ffn(inputs, expansion=4, kernel_size=1, drop_rate=0, activation="gelu", name=""):
    """x ← x + Module (Norm(x)), similar with typical MLP block"""
    # preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")
    preact = layer_norm(inputs, name=name + "preact_")

    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(preact, input_channel * expansion, kernel_size, name=name + "1_")
    nn = activation_by_name(nn, activation=activation, name=name)
    nn = conv2d_no_bias(nn, input_channel, kernel_size, name=name + "2_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # return keras.layers.Add(name=name + "output")([preact, nn])
    return keras.layers.Add(name=name + "output")([inputs, nn])


def res_mhsa(inputs, output_channel, conv_short_cut=True, strides=1, head_dimension=32, drop_rate=0, activation="gelu", name=""):
    """x ← Proj(Pool(x)) + Attention (Pool(Norm(x)))"""
    # preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")
    preact = layer_norm(inputs, name=name + "preact_")

    if conv_short_cut:
        shortcut = keras.layers.MaxPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
        # shortcut = batchnorm_with_activation(shortcut, activation=activation, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    nn = preact
    if strides != 1:  # Downsample
        # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.MaxPool2D(pool_size=2, strides=strides, padding="SAME", name=name + "pool")(nn)
    num_heads = nn.shape[-1] // head_dimension
    nn = mhsa_with_multi_head_relative_position_embedding(nn, num_heads=num_heads, key_dim=head_dimension, out_shape=output_channel, name=name + "mhsa_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{name = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return keras.layers.Add(name=name + "output")([shortcut, nn])


class MultiHeadRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, with_cls_token=True, attn_height=-1, num_heads=-1, **kwargs):
        super(MultiHeadRelativePositionalEmbedding, self).__init__(**kwargs)
        self.with_cls_token, self.attn_height, self.num_heads = with_cls_token, attn_height, num_heads
        if with_cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        # input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
        # input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
        # print(attn_shape)
        if self.attn_height == -1:
            height = width = int(tf.math.sqrt(float(attn_shape[2] - self.cls_token_len)))  # hh == ww, e.g. 14
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        # pos_shape = (num_relative_distance, num_heads)
        pos_shape = (num_heads, num_relative_distance)
        self.relative_position_bias_table = self.add_weight(name="positional_embedding", shape=pos_shape, initializer="zeros", trainable=True)

        hh, ww = tf.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = tf.stack([hh, ww], axis=-1)  # [14, 14, 2]
        coords_flatten = tf.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = tf.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [196, 196]
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.with_cls_token:
            top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            # print(f">>>> {top.shape = }, {left.shape = }, {corner.shape = }")
            # >>>> top.shape = TensorShape([1, 196]), left.shape = TensorShape([196, 1]), corner.shape = TensorShape([1, 1])
            left_corner = tf.concat([corner, left], axis=0)
            relative_position_index = tf.concat([top, relative_position_index], axis=0)
            relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [197, 197]
        self.relative_position_index = relative_position_index

    def call(self, attention_scores, **kwargs):
        pos_emb = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=1)
        # tf.print(pos_emb.shape, attention_scores.shape)
        return attention_scores + pos_emb

    def get_config(self):
        base_config = super(MultiHeadRelativePositionalEmbedding, self).get_config()
        base_config.update({"with_cls_token": self.with_cls_token, "attn_height": self.attn_height, "num_heads": self.num_heads})
        return base_config

    def load_resized_pos_emb(self, source_layer, method="nearest"):
        if isinstance(source_layer, dict):
            source_tt = source_layer["positional_embedding:0"]  # weights
            # source_tt = source_layer["pos_emb:0"]  # weights
        else:
            source_tt = source_layer.relative_position_bias_table  # layer
        # self.relative_position_bias_table.assign(tf.transpose(source_tt))
        hh = ww = int(tf.math.sqrt(float(source_tt.shape[1] - self.cls_token_pos_len)))  # assume source weights are all square shape
        num_heads = source_tt.shape[0]
        ss = tf.reshape(source_tt[:, : hh * ww], (num_heads, hh, ww))  # [num_heads, hh, ww]
        ss = tf.transpose(ss, [1, 2, 0])  # [hh, ww, num_heads]

        if self.attn_height == -1:
            target_hh = target_ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len)))
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len) / target_hh)
        tt = tf.image.resize(ss, [target_hh, target_ww], method=method)  # [target_hh, target_ww, num_heads]
        tt = tf.reshape(tt, (tt.shape[0] * tt.shape[1], num_heads))  # [target_hh * target_ww, num_heads]
        tt = tf.transpose(tt)  # [num_heads, target_hh * target_ww]
        if self.with_cls_token:
            tt = tf.concat([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
        self.relative_position_bias_table.assign(tt)

    def show_pos_emb(self, rows=1, base_size=2):
        import matplotlib.pyplot as plt

        hh = ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[0] - self.cls_token_pos_len)))
        ss = tf.reshape(self.relative_position_bias_table[: hh * ww], (hh, ww, -1)).numpy()
        cols = int(tf.math.ceil(ss.shape[-1] / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            ax.imshow(ss[:, :, id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig



def CoAtNet(
    num_blocks,
    out_channels,
    stem_width=64,
    block_types=["conv", "conv", "transform", "transform"],
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    use_dw_strides=True,
    bn_act_first=False,  # Experiment, use activation -> BatchNorm instead of BatchNorm -> activation, also set use_bias=True for pre Conv2D layer
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="coatnet",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, use_bias=bn_act_first, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, use_bias=bn_act_first, padding="same", name="stem_2_")
    # nn = batchnorm_with_activation(nn, activation=activation, name="stem_2_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        stack_strides = strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        for block_id in range(num_block):
            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 else False
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block:
                nn = res_MBConv(
                    nn, out_channel, conv_short_cut, stride, expansion, block_se_ratio, block_drop_rate, use_dw_strides, bn_act_first, activation, name=name
                )
            else:
                nn = res_mhsa(nn, out_channel, conv_short_cut, stride, head_dimension, block_drop_rate, activation=activation, name=name)
                nn = res_ffn(nn, expansion=expansion, drop_rate=block_drop_rate, activation=activation, name=name + "ffn_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation, act_first=bn_act_first)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "coatnet", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def CoAtNetT(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", **kwargs):
    num_blocks = [3, 4, 6, 3]
    out_channels = [64, 128, 256, 512]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnett", **kwargs)


def CoAtNet0(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 5, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet0", **kwargs)


def CoAtNet1(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.3, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet1", **kwargs)


def CoAtNet2(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.5, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [128, 256, 512, 1024]
    stem_width = 128
    return CoAtNet(**locals(), model_name="coatnet2", **kwargs)


def CoAtNet3(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.7, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet3", **kwargs)


def CoAtNet4(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet4", **kwargs)


def CoAtNet5(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [256, 512, 1280, 2048]
    stem_width = 192
    head_dimension = 64
    return CoAtNet(**locals(), model_name="coatnet5", **kwargs)


def CoAtNet6(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [192, 384, 768, 1536, 2048]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet6", **kwargs)


def CoAtNet7(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [256, 512, 1024, 2048, 3072]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet7", **kwargs)

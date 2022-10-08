import re
import numpy as np
from matplotlib import pyplot as plt
import os
import numpy as np
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
# sys.path.append('./eff_feature.py')
# sys.path.append('./swin_sp.py')
# sys.path.append("./Initializer.py")
#Modified efficient net to match the config of the timm
import tensorflow as tf
from . import eff_feature as efm
from . import swin_sp as swin
from . import Initializer
import tensorflow_addons as tfa
AUTO = tf.data.experimental.AUTOTUNE
import wandb
import gc
#------------------------------
from keras import backend
from keras.distribute import distributed_file_utils
from keras.distribute import worker_training_state
# from keras.optimizers.schedules import learning_rate_schedule
from keras.utils import generic_utils
from keras.utils import io_utils
from keras.utils import tf_utils
from keras.utils import version_utils
from keras.utils.data_utils import Sequence
from keras.utils.generic_utils import Progbar
from keras.utils.mode_keys import ModeKeys
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

try:
  import requests
except ImportError:
  requests = None


# eff_blocks = 2

# model_a = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 1)

# model_b = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 2)

# model_ef = efm.EfficientNetB3(input_shape=(224,224,8),feature_extractor_blocks=eff_blocks)

# model_top = tf.keras.layers.Dense(1, kernel_initializer=Initializer.MLP_Normal)

# feature_extractor_blocks=eff_blocks
# swin_blocks = model_b.num_layers - feature_extractor_blocks
# embeded_dim = model_a.embed_dim[0] * (2**(4 - swin_blocks))
# connector = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=embeded_dim,use_bias=False, kernel_size = 1),
#                                  tf.keras.layers.BatchNormalization(momentum = 0.1, epsilon = 1e-05),
#                                  tf.keras.layers.Activation("swish"),
#                                  tf.keras.layers.Reshape((-1,embeded_dim))
# ])

class Stage1(tf.keras.Model):

    def __init__(self):
        super().__init__()

        eff_blocks = 2
        self.model_a = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 1)
        self.model_b = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 2)
        self.top = tf.keras.layers.Dense(1, kernel_initializer=Initializer.MLP_Normal)


    def call(self, x):
        x = self.model_a(x)
        x = self.model_b(x)
        x = self.top(x)
        return x


class Stage2(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embed_dim = 96
        eff_blocks = 2
        self.model_b = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 2)
        self.eff = efm.EfficientNetB3(input_shape=(224,224,8), feature_extractor_blocks=eff_blocks)

        feature_extractor_blocks=eff_blocks
        swin_blocks = self.model_b.num_layers - feature_extractor_blocks
        embeded_dim = embed_dim * (2**(4 - swin_blocks))

        self.connector = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=embeded_dim,use_bias=False, kernel_size = 1),
                                         tf.keras.layers.BatchNormalization(momentum = 0.1, epsilon = 1e-05),
                                         tf.keras.layers.Activation("swish"),
                                         tf.keras.layers.Reshape((-1,embeded_dim))])


        self.model_b.trainable = False
        self.top = tf.keras.layers.Dense(1, kernel_initializer=Initializer.MLP_Normal)
        self.top.trainable = False

    def call(self, x):
        x = self.eff(x)
        x = self.connector(x)
        x = self.model_b(x)
        x = self.top(x)
        return x


class Stage3(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embed_dim = 96
        eff_blocks = 2
        self.model_b = swin.SwinTransformer(part_model  =True, eff_blocks = eff_blocks, partition_ = 2)
        self.eff = efm.EfficientNetB3(input_shape=(224,224,8),feature_extractor_blocks=eff_blocks)
        feature_extractor_blocks=eff_blocks
        swin_blocks = self.model_b.num_layers - feature_extractor_blocks
        embeded_dim = embed_dim * (2**(4 - swin_blocks))

        self.connector = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=embeded_dim,use_bias=False, kernel_size = 1),
                                         tf.keras.layers.BatchNormalization(momentum = 0.1, epsilon = 1e-05),
                                         tf.keras.layers.Activation("swish"),
                                         tf.keras.layers.Reshape((-1,embeded_dim))])

        self.model_b.trainable = True
        self.top = tf.keras.layers.Dense(1, kernel_initializer=Initializer.MLP_Normal)
        self.top.trainable = True
    def call(self, x):
        x = self.eff(x)
        x = self.connector(x)
        x = self.model_b(x)
        x = self.top(x)
        return x

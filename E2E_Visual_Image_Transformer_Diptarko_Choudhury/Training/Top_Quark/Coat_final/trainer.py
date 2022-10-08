import tensorflow as tf
print("Tensorflow version " + tf.__version__)
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)


import re
import numpy as np
from matplotlib import pyplot as plt
import os
import numpy as np
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
sys.path.append('./coat.py')
import coat
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



wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
wandb.init(
    project = "Top_CoAt_Final"
)

BATCH_SIZE = 128 * tpu_strategy.num_replicas_in_sync
SHUFFLE_BUFFER = 2048*6
LR = 5e-4
WD = 1e-5
LR_2 = 1e-3
WD_2 = 5e-5
ROT_ANGLE = 60
GCS_DS_PATH = "gs://top_dataset_us/TFR"
VAL_STEPS = 32*4
TRAIN_STEPS = 194*4*5
EPOCHS = 100
WARMUP_EPOCHS = 1
BATCH_SHUFFLE_BUFFER = 10
wandb.config.update({"learning_rate_stage1":LR,
                    "weight_decay_stage1":WD,
                    "learning_rate_stage2":LR_2,
                    "weight_decay_stage2":WD_2,
                    "Rotation Angle":ROT_ANGLE,
                    "Shuffle_buffer size":SHUFFLE_BUFFER,
                    "BATCH_SIZE":BATCH_SIZE,
                    "Validation steps":VAL_STEPS,
                    "Training Steps":TRAIN_STEPS,
                    "Training Epochs":EPOCHS,
                    "WarmUp_Epochs":WARMUP_EPOCHS,
                    "BATCH_SHUFFLE_BUFFER":BATCH_SHUFFLE_BUFFER
                    })

FILES = tf.io.gfile.glob(GCS_DS_PATH + '/*.tfrecords')
TRAIN = FILES[:250]
TEST = FILES[250:300]


def _parse_image_label_function(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'channel_1': tf.io.FixedLenFeature([], tf.string),
        'channel_2': tf.io.FixedLenFeature([], tf.string),
        'channel_3': tf.io.FixedLenFeature([], tf.string),
        'channel_4': tf.io.FixedLenFeature([], tf.string),
        'channel_5': tf.io.FixedLenFeature([], tf.string),
        'channel_6': tf.io.FixedLenFeature([], tf.string),
        'channel_7': tf.io.FixedLenFeature([], tf.string),
        'channel_8': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string)}

    # Parse the input tf.train.Example proto using the dictionary above.
    content = tf.io.parse_single_example(example_proto, image_feature_description)
    content["channel_1"] = tf.io.decode_png(content["channel_1"], channels=1)
    content["channel_2"] = tf.io.decode_png(content["channel_2"], channels=1)
    content["channel_3"] = tf.io.decode_png(content["channel_3"], channels=1)
    content["channel_4"] = tf.io.decode_png(content["channel_4"], channels=1)
    content["channel_5"] = tf.io.decode_png(content["channel_5"], channels=1)
    content["channel_6"] = tf.io.decode_png(content["channel_6"], channels=1)
    content["channel_7"] = tf.io.decode_png(content["channel_7"], channels=1)
    content["channel_8"] = tf.io.decode_png(content["channel_8"], channels=1)
    content["channel_1"] = tf.cast(content["channel_1"], tf.float32) / 255.0
    content["channel_2"] = tf.cast(content["channel_2"], tf.float32) / 255.0
    content["channel_3"] = tf.cast(content["channel_3"], tf.float32) / 255.0
    content["channel_4"] = tf.cast(content["channel_4"], tf.float32) / 255.0
    content["channel_5"] = tf.cast(content["channel_5"], tf.float32) / 255.0
    content["channel_6"] = tf.cast(content["channel_6"], tf.float32) / 255.0
    content["channel_7"] = tf.cast(content["channel_7"], tf.float32) / 255.0
    content["channel_8"] = tf.cast(content["channel_8"], tf.float32) / 255.0
    label = content["label"]
    return tf.concat([ content["channel_1"],
                       content["channel_2"],
                       content["channel_3"],
                       content["channel_4"],
                       content["channel_5"],
                       content["channel_6"],
                       content["channel_7"],
                       content["channel_8"]],axis = -1),label

def data_augment_train(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.resize(image, (128,128), method = "bicubic")
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    rot_angle = ROT_ANGLE
    image = tfa.image.rotate(image,rot_angle*(np.random.rand()-0.5)*2)
    return image, label


def data_augment_test(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.resize(image, (128,128), method = "bicubic")
    return image, label


def load_dataset(filenames,train = False):
#     ignore_order = tf.data.Options()
#     ignore_order.experimental_deterministic = False
#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
#     dataset = dataset.with_options(ignore_order)

    dataset = tf.data.TFRecordDataset.list_files(filenames)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=300,
        num_parallel_calls=AUTO,
        deterministic=False,
        block_length=1)
    dataset = dataset.map(_parse_image_label_function, num_parallel_calls=AUTO)
    if train:
        dataset = dataset.map(data_augment_train, num_parallel_calls = AUTO)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.shuffle(BATCH_SHUFFLE_BUFFER)
    else:
        dataset = dataset.map(data_augment_test, num_parallel_calls=AUTO)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


    dataset = dataset.prefetch(AUTO)

    return dataset

train_ds = load_dataset(TRAIN, train = True)
val_ds = load_dataset(TEST, train = False)


class auc_sklearn(tf.keras.callbacks.Callback):

    def __init__(self, val_ds, val_steps):
        super(auc_sklearn, self).__init__()
        self.val_ds = val_ds
        self.val_steps = val_steps
        self.auc_hist =[]
        self.auc_last = 0
        self.loss_last = 0
        self.loss_hist = []
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))




    def on_epoch_end(self, epoch, logs=None):

        @tf.function
        def valid_step(images, labels):
            probabilities = model(images, training=False)
            loss = loss_fn(labels, probabilities)
            valid_loss.update_state(loss)
            return labels,probabilities

        i = 0
        step = self.val_steps
        l = np.array([], dtype=np.int64).reshape(0)
        p = np.array([], dtype=np.float32).reshape(0)
        for image, labels in self.val_ds:
            labels,probability = tpu_strategy.run(valid_step, args=(image, labels))
            for j in range(tpu_strategy.num_replicas_in_sync):
                l = np.concatenate([l,labels.values[j].numpy()])
                p = np.concatenate([p,tf.squeeze(probability.values[j]).numpy()])
            i = i+1
            if i == step:
                break
        auc = roc_auc_score(l,p)
        print(f"{epoch+1} Epoch Ended, Val_loss:{valid_loss.result().numpy()/self.val_steps}, Val_auc:{auc}")

        self.auc_hist.append(auc)
        self.auc_last = auc
        self.loss_last = valid_loss.result().numpy()/self.val_steps
        self.loss_hist.append(self.loss_last)
        valid_loss.reset_states()


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
  """Hardcoded DO NOT CHANGE  """

  def __init__(self,
               warmup_epochs,
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    super(ReduceLROnPlateau, self).__init__()

    if factor >= 1.0:
      raise ValueError(
          f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')
    if 'epsilon' in kwargs:
      min_delta = kwargs.pop('epsilon')
      logging.warning('`epsilon` argument is deprecated and '
                      'will be removed, use `min_delta` instead.')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self.warmup_epochs = warmup_epochs
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if self.mode not in ['auto', 'min', 'max']:
      logging.warning('Learning rate reduction mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = backend.get_value(self.model.optimizer.lr)
    current = auc.auc_last
    if epoch == self.warmup_epochs:
        print("Warmup Completed Decay started")
    if epoch >= self.warmup_epochs:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = backend.get_value(self.model.optimizer.lr)
          old_wd = backend.get_value(self.model.optimizer.weight_decay)
          if old_lr > np.float32(self.min_lr):
            new_lr = old_lr * self.factor
            new_wd = old_wd * self.factor
            new_lr = max(new_lr, self.min_lr)
            backend.set_value(self.model.optimizer.lr, new_lr)
            backend.set_value(self.model.optimizer.weight_decay, new_wd)
            if self.verbose > 0:
              print(f'\nEpoch {epoch +1}:ReduceLROnPlateau reducing learning rate to {new_lr}., weight_decay to {new_wd}')
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0


class LinearWarmUp(tf.keras.callbacks.Callback):
  """Hardcoded DO NOT CHANGE  """

  def __init__(self,
               steps_per_epoch,
               warmup_epochs,
               max_lr,
               max_wd,
               min_lr,
               min_wd
              ):
    super(LinearWarmUp, self).__init__()

    self.steps_per_epoch = steps_per_epoch
    self.warmup_epochs = warmup_epochs
    self.max_lr = max_lr
    self.min_lr = min_lr
    self.max_wd = max_wd
    self.min_wd = min_wd
    self.total_steps = self.steps_per_epoch*self.warmup_epochs
    self.step = 0
    self.curr_lr = self.min_lr
    self.curr_wd = self.min_wd
    self.delta_lr = 0
    self.delta_wd = 0

  def _reset(self):
    self.step = 0
    self.delta_lr = (self.max_lr - self.min_lr)/(self.total_steps)
    self.delta_wd = (self.max_wd - self.min_wd)/(self.total_steps)
    self.curr_lr = self.min_lr
    self.curr_wd = self.min_wd

  def on_train_begin(self, logs=None):
    self._reset()
    print("WarmUp started")

  def on_train_batch_end(self, batch, logs=None):
    self.step = self.step + 1
    if self.step < self.total_steps:
        new_lr = self.min_lr + (self.step)*self.delta_lr
        new_wd = self.min_wd + (self.step)*self.delta_wd
        self.curr_lr = new_lr
        self.curr_wd = new_wd
        backend.set_value(self.model.optimizer.lr, new_lr)
        backend.set_value(self.model.optimizer.weight_decay, new_wd)


class logger(tf.keras.callbacks.Callback):
    def __init__(self):
      super(logger, self).__init__()

    def on_epoch_end(self,epoch, logs=None):
        wandb.log({"AUC": logs["auc"],
                   "Epoch": epoch,
                   "Val_auc": auc.auc_last,
                    "Val_loss":auc.loss_last,
                   "loss": logs["loss"],
                   "learning_rate": backend.get_value(self.model.optimizer.lr),
                   "weight_decay": backend.get_value(self.model.optimizer.weight_decay)})


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, target):
        '''
        Tensorflow Callback to only save the best weights which are above
        certain threshold.
        args:
        target:The minimum value above which the checkpoints will be saved.
        '''
        super(Checkpoint, self).__init__()
        self.best = target

    def on_epoch_end(self, epoch, logs=None):
        if auc.auc_last >= self.best:
            self.best = auc.auc_last
            wandb.alert(title="Crossed the target",
                        text=f"The current validation auc is {auc.auc_last}")
            with tpu_strategy.scope():
                checkpoint = tf.train.Checkpoint(model=model)
                checkpoint.save("./checkpoint/Weight")




with tpu_strategy.scope():

    model = coat.CoAtNet(
    num_blocks = [2, 3, 5, 2],
    out_channels = [96, 192, 384, 768],
    stem_width=64,
    block_types=["conv", "conv", "transform", "transform"],
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    use_dw_strides=True,
    bn_act_first=False,  # Experiment, use activation -> BatchNorm instead of BatchNorm -> activation, also set use_bias=True for pre Conv2D layer
    input_shape=(128, 128, 8),
    num_classes=1,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="linear",
    dropout=0,
    pretrained=None,
    model_name="coatnet")

    model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD),
        # steps_per_execution = 32,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy", "AUC"])

    valid_loss = tf.keras.metrics.Sum(name="val_accuracy")
    loss_fn = lambda a,b: tf.nn.compute_average_loss(
        tf.keras.losses.binary_crossentropy(tf.reshape(a, (-1,1)),b, from_logits=True), global_batch_size=BATCH_SIZE)



auc = auc_sklearn(val_ds=tpu_strategy.experimental_distribute_dataset(val_ds), val_steps=VAL_STEPS)
lr_sc = ReduceLROnPlateau(
    monitor=auc.auc_last,
    factor=0.5,
    patience=5,
    verbose=1,
    mode='max',
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-7,
    warmup_epochs=WARMUP_EPOCHS)

lr_warm = LinearWarmUp(steps_per_epoch=TRAIN_STEPS,
                       warmup_epochs=WARMUP_EPOCHS,
                       max_lr=LR,
                       max_wd=WD,
                       min_lr=0,
                       min_wd=0)
log = logger()
checkpoint = Checkpoint(target=0.9872)

history = model.fit(train_ds,
                    steps_per_epoch=TRAIN_STEPS,
                    callbacks=[auc, lr_warm, lr_sc, log, checkpoint],
                    epochs=EPOCHS
                    )

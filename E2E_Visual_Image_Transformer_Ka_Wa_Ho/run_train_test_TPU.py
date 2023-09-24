import sys
import wandb
import argparse
import numpy as np

from models.swins import SwinTransformer

from keras import backend
from sklearn.metrics import roc_auc_score
from tensorflow.python.platform import tf_logging as logging

import tensorflow as tf
import tensorflow_addons as tfa

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser(description='Script to run training of ViT models')
parser.add_argument('-p', '--patch', type=int, default=2, help='patch size (default: %(default)s)')
parser.add_argument('-w', '--win', type=str, default='4', help='window size separated by commas (default: %(default)s)')
parser.add_argument('-e', '--emb', type=int, default=96, help='embedding dimension (default: %(default)s)')
parser.add_argument('-h', '--head', type=str, default='3,6,12,24', help='number of heads in the multi-head attention in the different layers separated by commas (default: %(default)s)')
parser.add_argument('-sw', action='store_true', help='use shifting window operation as in Swin')
parser.add_argument('-cw', action='store_true', help='use layer-wise convolution as in Win')
args = parser.parse_args()

args.head = (int(_) for _ in args.head.split(','))
args.w = (int(_) for _ in args.w.split(','))

wandb.login()

config = {
    "batch_size": 32*tpu_strategy.num_replicas_in_sync,
    'shuff_buf': 2048*6,
    'lr': 5e-5,
    'wd': 5e-6,
}

wandb.init(
    project=f"Vision_Transformers",
    name="Swin",
    config=config,
#    mode="disabled"
)

files_train = tf.io.gfile.glob('data/top/tf/*train*.tfrecords')
files_valid = tf.io.gfile.glob('data/top/tf/*valid*.tfrecords')
files_test = tf.io.gfile.glob('data/top/tf/*test*.tfrecords')

def _parse_image_label_function(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string)
    }

    for _ in range(8):
        image_feature_description[f'ch{_}'] = tf.io.FixedLenFeature([], tf.string)

    # Parse the input tf.train.Example proto using the dictionary above.
    content = tf.io.parse_single_example(example_proto, image_feature_description)
    for _ in range(8):
      content[f"ch{_}"] = tf.cast(tf.io.decode_png(content[f"ch{_}"], channels=1), tf.float32) / 255.0

    label = content["label"]
    return tf.concat([content[f"ch{i}"] for i in range(8)], axis = -1), label


def data_preprocess(image, label):
    image = tf.image.pad_to_bounding_box(image, 2, 2, 128, 128)
    return image, label

def data_augment(image, label):
    image = tf.image.pad_to_bounding_box(image, 2, 2, 128, 128)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tfa.image.rotate(image, 10*np.pi*(np.random.rand()-0.5)*2/180)
    image = tfa.image.translate(image, [(np.random.rand()-0.5)*2*128*0.04, 0])
    return image, label

def load_dataset(filenames, train):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset.list_files(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=300,
        num_parallel_calls=AUTO,
        deterministic=False,
        block_length=1)
    dataset = dataset.map(_parse_image_label_function, num_parallel_calls=AUTO)
    if train:
        dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
        dataset = dataset.shuffle(config['shuff_buf'])
        dataset = dataset.batch(config['batch_size'], drop_remainder=True)

    else:
        dataset = dataset.map(data_preprocess, num_parallel_calls=AUTO)
        dataset = dataset.batch(config['batch_size'], drop_remainder=True)

    dataset = dataset.prefetch(AUTO)

    return dataset

train_ds = load_dataset(files_train, train=True)
valid_ds = load_dataset(files_valid, train=False)
test_ds = load_dataset(files_test, train=False)

class auc_sklearn(tf.keras.callbacks.Callback):

    def __init__(self, eval_ds, save_model=False):
        super(auc_sklearn, self).__init__()
        self.eval_ds = eval_ds
        self.auc_last = 0
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs=None):

        @tf.function
        def valid_step(images, labels):
            probs = model(images, training=False)
            return labels, probs

        l, p = [], []
        for image, labels in self.eval_ds:
            labels, probs = tpu_strategy.run(valid_step, args=(image, labels))
            for _ in range(tpu_strategy.num_replicas_in_sync):
                l.append(labels.values[_].numpy())
                p.append(tf.math.sigmoid(tf.squeeze(probs.values[_])).numpy())
        auc = roc_auc_score(np.concatenate(l), np.concatenate(p))
        print(f"{epoch+1} Epoch Ended, auc:{auc}")

        if self.save_model:
            model.save_weights(f"model_{epoch}.weights.h5") 

        self.auc_last = auc

class ReduceLROnPlateau(tf.keras.callbacks.Callback):
  """Hardcoded DO NOT CHANGE  """

  def __init__(self,
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
    current = auc_valid.auc_last
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
          new_lr = max(new_lr, self.min_lr])
          backend.set_value(self.model.optimizer.lr, new_lr)
          backend.set_value(self.model.optimizer.weight_decay, new_wd)
          if self.verbose > 0:
            print(f'\nEpoch {epoch +1}:ReduceLROnPlateau reducing learning rate to {new_lr}., weight_decay to {new_wd}')
          self.cooldown_counter = self.cooldown
          self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0

auc_valid = auc_sklearn(tpu_strategy.experimental_distribute_dataset(valid_ds))
auc_test = auc_sklearn(tpu_strategy.experimental_distribute_dataset(test_ds), save_model=True)

lr_sc = ReduceLROnPlateau(
            monitor=auc_valid.auc_last,
            factor=0.5,
            patience=3,
            verbose=1,
            mode='max',
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-7,
        )

class logger(tf.keras.callbacks.Callback):
    def __init__(self):
      super(logger, self).__init__()

    def on_epoch_end(self,epoch, logs=None):
        wandb.log({
            "Epoch": epoch,
            "Test_auc_epoch": auc_valid.auc_last,
            "lr": backend.get_value(self.model.optimizer.lr),
            "weight_decay": backend.get_value(self.model.optimizer.weight_decay)
        })

cfg = dict(
    img_size=128,
    patch_size=args.patch,
    window_size=args.w,
    embed_dim=args.emb,
    depths=(2, 2, 6, 2),
    num_heads=args.head,
    num_classes=1,
    shift_win=args.sw,
    conv_win=args.cw
)

with tpu_strategy.scope():
    model = SwinTransformer(name="swin", **cfg)
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=config['lr'], weight_decay=config['wd']),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0), 
            tf.keras.metrics.AUC(from_logits=True)
            ]
        )

history = model.fit(train_ds,
                    callbacks=[auc_valid, auc_test, lr_sc, logger()],
                    epochs=100
                    )

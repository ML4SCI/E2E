import tensorflow as tf

class Kaimming_Uniform(tf.keras.initializers.VarianceScaling):
    def __init__(self, seed=None):
        super(Kaimming_Uniform, self).__init__(
        scale=0.3,
        mode='fan_in',
        distribution='uniform',
        seed=seed)

    def get_config(self):
        return {'seed': self.seed}

class Kaimming_Normal(tf.keras.initializers.VarianceScaling):
    def __init__(self, seed=None):
        super(Kaimming_Normal, self).__init__(
        scale=0.3,
        mode='fan_avg',
        distribution='normal',
        seed=seed)

    def get_config(self):
        return {'seed': self.seed}

class Relative_pos_b_t(tf.keras.initializers.TruncatedNormal):
    def __init__(self, seed = None):
        super(Relative_pos_b_t, self).__init__(
        mean = 0,
        stddev=0.02,
        seed = seed)

    def get_config(self):
        return {'seed': self.seed}

class MLP_Normal(tf.keras.initializers.RandomNormal):
    def __init__(self, seed = None):
        super(MLP_Normal, self).__init__(
        mean = 0,
        stddev=0.02,
        seed = seed)
    def get_config(self):
        return {'seed': self.seed}

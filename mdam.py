import tensorflow as tf


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // reduction_ratio, activation='elu'),
            tf.keras.layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, inputs):
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        channel_weights = self.mlp(max_pool)
        return channel_weights


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[2, 3], keepdims=True)
        temporal_weights = self.conv1d(avg_pool)
        return temporal_weights


class FrequencyAttention(tf.keras.layers.Layer):
    def __init__(self, frequency_dim, reduction_ratio=16, **kwargs):
        super(FrequencyAttention, self).__init__(**kwargs)
        self.frequency_dim = frequency_dim
        self.reduction_ratio = reduction_ratio
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(1 // reduction_ratio, activation='elu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        max_pool = tf.reduce_max(inputs, axis=[1, 3], keepdims=True)
        frequency_weights = self.mlp(max_pool)
        frequency_weights = tf.keras.layers.Reshape((1, inputs.shape[2], 1))(frequency_weights)
        return frequency_weights


class MultiDimensionalAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super(MultiDimensionalAttention, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.temporal_attention = TemporalAttention(kernel_size)
        self.frequency_attention = FrequencyAttention(reduction_ratio)
        self.alpha = tf.Variable(1.0, trainable=True)
        self.beta = tf.Variable(1.0, trainable=True)
        self.gamma = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        channel_weights = self.channel_attention(inputs)
        temporal_weights = self.temporal_attention(inputs)
        frequency_weights = self.frequency_attention(inputs)

        S_c = tf.multiply(inputs, channel_weights)
        S_t = tf.multiply(inputs, temporal_weights)
        S_f = tf.multiply(inputs, frequency_weights)

        # Normalize the weights
        weights_sum = self.alpha + self.beta + self.gamma
        alpha_norm = self.alpha / weights_sum
        beta_norm = self.beta / weights_sum
        gamma_norm = self.gamma / weights_sum

        S_prime = alpha_norm * S_c + beta_norm * S_t + gamma_norm * S_f
        return S_prime

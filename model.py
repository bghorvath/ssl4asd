import numpy as np
import tensorflow as tf
from mixup_layer import MixupLayer
from openl3_idea_aug_layer_classwise import AugLayer
from subcluster_adacos import SCAdaCos
from statex_aug_layer_classwise import StatExLayer
from mdam import MultiDimensionalAttention


class SqueezeAndExcitationBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, ratio=16, dimension=2, **kwargs):
        super(SqueezeAndExcitationBlock, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.ratio = ratio
        self.dimension = dimension
        if self.dimension==2:
            self.L1 = tf.keras.layers.GlobalAveragePooling2D()
        elif self.dimension == 1:
            self.L1 = tf.keras.layers.GlobalAveragePooling1D()
        self.L2 = tf.keras.layers.Dense(self.num_channels//self.ratio, activation='relu', use_bias=False)
        self.L3 = tf.keras.layers.Dense(self.num_channels, activation='sigmoid', use_bias=False)
        self.L4 = tf.keras.layers.Multiply()


    def build(self, input_shape):
        super(SqueezeAndExcitationBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.L1(inputs)
        x = self.L2(x)
        x = self.L3(x)
        return self.L4([inputs, x])

    def get_config(self):
        config = {
            'num_channels': self.num_channels,
            'ratio': self.ratio,
            'dimension': self.dimension
        }
        config.update(super(SqueezeAndExcitationBlock, self).get_config())
        return config


def adjust_size(wav, new_size):
    reps = int(np.ceil(new_size/wav.shape[0]))
    offset = np.random.randint(low=0, high=int(reps*wav.shape[0]-new_size+1))
    return np.tile(wav, reps=reps)[offset:offset+new_size]


class MagnitudeSpectrogram(tf.keras.layers.Layer):
    """
    Compute magnitude spectrograms.
    https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
    """

    def __init__(self, sample_rate, fft_size, hop_size, f_min=0.0, f_max=None, **kwargs):
        super(MagnitudeSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2

    def build(self, input_shape):
        super(MagnitudeSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)
        magnitude_spectrograms = tf.abs(spectrograms)
        magnitude_spectrograms = tf.expand_dims(magnitude_spectrograms, 3)
        return magnitude_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(MagnitudeSpectrogram, self).get_config())
        return config

def mixupLoss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=y_pred[:, :, 1], y_pred=y_pred[:, :, 0])


def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def model_emb_cnn(num_classes, raw_dim, n_subclusters, use_bias=False, mdam=False, mixup=False, statex=False, featex=False):
    data_input = tf.keras.layers.Input(shape=(raw_dim, 1), dtype='float32')
    label_input = tf.keras.layers.Input(shape=(num_classes), dtype='float32')
    y = label_input
    x = data_input
    l2_weight_decay = tf.keras.regularizers.l2(1e-5)
    x_mix, y_mix = x, y
    if mixup:
        x_mix, y_mix = MixupLayer(prob=0.5)([x, y])

    # FFT
    x = tf.keras.layers.Lambda(lambda x: tf.math.abs(tf.signal.fft(tf.complex(x[:, :, 0], tf.zeros_like(x[:, :, 0])))[:, :int(raw_dim / 2)]))(x_mix)
    x = tf.keras.layers.Reshape((-1,1))(x)
    x = tf.keras.layers.Conv1D(128, 256, strides=64, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)
    x = SqueezeAndExcitationBlock(num_channels=128, dimension=1)(x)
    x = tf.keras.layers.Conv1D(128, 64, strides=32, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)
    x = SqueezeAndExcitationBlock(num_channels=128, dimension=1)(x)
    x = tf.keras.layers.Conv1D(128, 16, strides=4, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)
    x = SqueezeAndExcitationBlock(num_channels=128, dimension=1)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    emb_fft = tf.keras.layers.Dense(128, name='emb_fft', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)

    # magnitude
    x = tf.keras.layers.Reshape((raw_dim,))(x_mix)
    x = MagnitudeSpectrogram(16000, 1024, 512, f_max=8000, f_min=200)(x)

    if statex:
        x, y = StatExLayer(prob=0.5)([x,y_mix])

    x = tf.keras.layers.Lambda(lambda x: x-tf.math.reduce_mean(x, axis=1, keepdims=True))(x) # CMN-like normalization
    x = tf.keras.layers.BatchNormalization(axis=-2)(x)

    # first block
    x = tf.keras.layers.Conv2D(16, 7, strides=2, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    # second block
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=16)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=16)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # third block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=32)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=32)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # fourth block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=64)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=64)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # MDAM
    if mdam:
        x = MultiDimensionalAttention(channels=64)(x)

    # fifth block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=128)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay,
                                use_bias=use_bias)(xr)
    xr = SqueezeAndExcitationBlock(num_channels=128)(xr)
    x = tf.keras.layers.Add()([x, xr])

    x = tf.keras.layers.MaxPooling2D((18, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='flat')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    emb_mel = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, name='emb_mel', use_bias=use_bias)(x)
    
    emb_mel_ssl, emb_fft_ssl, y_ssl = emb_mel, emb_fft, y
    if featex:
        emb_mel_ssl, emb_fft_ssl, y_ssl = AugLayer(prob=0.5)([emb_mel,emb_fft,y])

    # prepare output
    x = tf.keras.layers.Concatenate(axis=-1)([emb_fft, emb_mel])
    x_ssl = tf.keras.layers.Concatenate(axis=-1)([emb_fft_ssl, emb_mel_ssl])

    output_ssl2 = SCAdaCos(n_classes=num_classes*9, n_subclusters=n_subclusters, trainable=True)([x_ssl, y_ssl, label_input])
    output = SCAdaCos(n_classes=num_classes, n_subclusters=n_subclusters, trainable=False)([x, y_mix, label_input])
    output_ssl = SCAdaCos(n_classes=num_classes*3, n_subclusters=n_subclusters, trainable=True)([x, y, label_input])
    loss_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output, y_mix])
    loss_output_ssl = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output_ssl, y])
    loss_output_ssl2 = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output_ssl2, y_ssl])

    return data_input, label_input, loss_output, loss_output_ssl, loss_output_ssl2

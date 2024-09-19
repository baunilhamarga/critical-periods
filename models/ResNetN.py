import argparse

import tensorflow as tf
import tensorflow.keras as keras

parser = argparse.ArgumentParser()
parser.add_argument("--resnet_n", default=9, type=int, help="n from Resnet paper.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class ResNet(keras.Model):
    class ResidualBlock(tf.Module):
        def __init__(self, filters: int, down_sample: bool):
            super().__init__()
            self.filters = filters
            self.down_sample = down_sample

        def __call__(self, x):
            out = x

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1) if not self.down_sample else (2, 2),
                                      padding="same",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.BatchNormalization()(out)
            out = keras.layers.ReLU()(out)

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.BatchNormalization()(out)

            if self.down_sample:
                residual = keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(2, 2),
                                               padding="same",
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.HeNormal)(x)
                residual = tf.keras.layers.BatchNormalization()(residual)
            else:
                residual = x

            out = out + residual
            out = keras.layers.ReLU()(out)
            return out

    def __init__(self, args):
        inputs = keras.layers.Input(shape=(32, 32, 3), dtype=tf.float32)
        outputs = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(
            inputs)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.ReLU()(outputs)

        for _ in range(0, args.resnet_n):
            outputs = self.ResidualBlock(16, False)(outputs)

        outputs = self.ResidualBlock(32, True)(outputs)
        for _ in range(1, args.resnet_n):
            outputs = self.ResidualBlock(32, False)(outputs)

        outputs = self.ResidualBlock(64, True)(outputs)
        for _ in range(1, args.resnet_n):
            outputs = self.ResidualBlock(64, False)(outputs)

        outputs = keras.layers.GlobalAveragePooling2D()(outputs)
        outputs = keras.layers.Dense(10, activation=tf.nn.softmax)(outputs)
        super().__init__(inputs, outputs)
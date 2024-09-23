import sys
import tensorflow as tf
import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.utils as keras_utils
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics._classification import accuracy_score

import sys
sys.path.append('../')

from utils import custom_functions as func

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name=''):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='Conv2D_{}'.format(name))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm1_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act1_{}'.format(name))(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm2_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act2_{}'.format(name))(x)
        x = conv(x)
    return x


def ResNet(input_shape, depth_block, iter=0, num_classes=10):
    num_filters = 16

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        num_res_blocks = depth_block[stack]
        for res_block in range(num_res_blocks):
            layer_name = str(stack)+'_'+str(res_block)+'_'+str(iter)
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             name=layer_name+'_1')
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             name=layer_name+'_2')
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=layer_name+'_3')
            x = keras.layers.add([x, y],
                                 name='Add_'+layer_name)
            x = Activation('relu', name='Actoutput'+layer_name)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define a function to dynamically select and create models
def build_model(model_type, input_shape=(32, 32, 3), num_classes=10, compile=False, N_layers=44):
    if (N_layers - 2) % 3 != 0:
        raise ValueError(f"Invalid number of layers {N_layers} for ResNet. (N_layers - 2) must be divisible by 3.")
    
    n = (N_layers - 2) / 3
    
    if model_type.startswith('ResNet'):
        model = ResNet(input_shape=input_shape, depth_block = 3 * [int(n)], num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    
    # Compile the model
    if compile:
        lr = 0.01
        sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    n_classes = np.max(y_train)+1
    y_train = np.eye(n_classes)[y_train].squeeze()
    y_test =  np.eye(n_classes)[y_test].squeeze()

    model = ResNet(X_train[0].shape, [7, 7, 7], num_classes=10)

    model = func.finetuning(model, X_train, y_train, X_test, y_test)
    final_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
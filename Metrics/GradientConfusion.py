import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import gen_batches
import tensorflow as tf
from scipy.spatial import distance

import sys

loss_object = tf.keras.losses.CategoricalCrossentropy()

def load_model(architecture_file='', weights_file=''):
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _relu6(x):
        return K.relu(x, max_value=6)

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        with keras.utils.CustomObjectScope({'relu6': _relu6,
                                'DepthwiseConv2D': layers.DepthwiseConv2D,
                               '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model

def cifar_resnet_data(debug=True, validation_set=False):
    print('Debuging Mode') if debug is True else print('Real Mode')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    #y_test = np.argmax(y_test, axis=1)

    if validation_set is False:
        return x_train, y_train, x_test, y_test
    else:
        datagen = generate_data_augmentation(x_train)
        for x_val, y_val in datagen.flow(x_train, y_train, batch_size=5000):
            break
        return x_train, y_train, x_test, y_test, x_val, y_val

def gradient(x, y, model):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_hat = model(x)
        loss = loss_object(y, y_hat)

    # Get the gradients of the loss w.r.t to the input image.
    #return tape.gradient(loss, x)
    return tape.gradient(loss, model.trainable_weights)

def subsampling(X_train, y_train, p=0.1):
    # Subsample 10% of the training data to avoid memory constraints
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=0)
    placeholder = np.zeros(X_train.shape[0])
    for train_index, _ in sss.split(placeholder, y_train):

        X_train_10p = np.zeros((len(train_index), X_train.shape[1], X_train.shape[2], X_train.shape[3]), np.float)
        y_train_10p = np.zeros((len(train_index), y_train.shape[1]))

        for i in range(0, len(train_index)):
            X_train_10p[i] = X_train[train_index[i]]
            y_train_10p[i] = y_train[train_index[i]]
        break

    return X_train_10p, y_train_10p

if __name__ == '__main__':
    np.random.seed(12227)

    X_train, y_train, X_test, y_test = cifar_resnet_data(debug=False)


    architecture_name = 'E:/Projects/Pruning/ScratchUnpruned/ResNet56_RandomInicialization'
    weights = 'E:/Projects/Pruning/ScratchUnpruned/ResNet56_RandomInicialization'
    model = load_model(architecture_name, weights)

    X_train, y_train = subsampling(X_train, y_train, p=0.01)
    for ep in range(1, 100):

        weights = 'E:/Projects/Pruning/ScratchUnpruned/ResNet56_epoch[{}].h5'.format(ep)
        model.load_weights(weights)

        grad_batch = []

        for batch in gen_batches(X_train.shape[0], 128):
            grad = gradient(tf.convert_to_tensor(X_train[batch], tf.float32), tf.convert_to_tensor(y_train[batch]), model)
            tmp = grad[0].numpy().reshape(-1)

            for i in range(1, len(grad)):
                tmp = np.hstack((tmp, grad[i].numpy().reshape(-1)))

            grad_batch.append(tmp)

        cosine = []
        for i in range(0, len(grad_batch)):
            for j in range(i+1, len(grad_batch)):
                cosine.append(distance.cosine(grad_batch[i], grad_batch[j]))
        print(min(cosine))

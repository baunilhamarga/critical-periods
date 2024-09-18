import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import gen_batches
import time
import tensorflow as tf
from scipy.spatial.distance import cosine
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)
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

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean

    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    #y_test = np.argmax(y_test, axis=1)

    if validation_set is False:
        return X_train, y_train, X_test, y_test
    else:
        datagen = generate_data_augmentation(X_train)
        for x_val, y_val in datagen.flow(X_train, y_train, batch_size=5000):
            break
        return X_train, y_train, X_test, y_test, x_val, y_val

def gradient(x, y, model):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_hat = model(x)
        loss = loss_object(y, y_hat)

    # Get the gradients of the loss w.r.t to the input image.
    #return tape.gradient(loss, x)
    return tape.gradient(loss, model.trainable_weights)

def get_kernel_layer_names(model):
    layer_names = []
    for l in model.layers:
        if len(l.weights) >0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

if __name__ == '__main__':
    np.random.seed(12227)

    X_train, y_train, X_test, y_test = cifar_resnet_data(debug=False)


    architecture_name = 'C:/Users/Artur/Desktop/residuals/E/Pruning/ScratchUnpruned/ResNet56_RandomInicialization'
    weights = 'C:/Users/Artur/Desktop/residuals/E/Pruning/ScratchUnpruned/ResNet56_RandomInicialization'
    model = load_model(architecture_name, weights)

    layer_names = get_kernel_layer_names(model)

    w_0 = [model.get_layer(l).get_weights()[0] for l in layer_names]

    cos_ep = []
    for ep in range(1, 100):
        weights = 'C:/Users/Artur/Desktop/residuals/E/Pruning/ScratchUnpruned/ResNet56_epoch[{}].h5'.format(ep)

        model.load_weights(weights)
        w_i = [model.get_layer(l).get_weights()[0] for l in layer_names]

        cos_layer = np.zeros((len(w_0)))

        for layer in range(0, len(w_0)):
            cos_layer[layer] = cosine(w_0[layer].flatten(), w_i[layer].flatten())

        print(cos_layer)#Print list in a single line
        #cos_ep.append(cos_layer)

    #print(cos_ep)
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from sklearn.utils import gen_batches
import os
import argparse

import sys
sys.path.append('../')

from utils import custom_functions as func
from models import custom_models

loss_object = tf.keras.losses.CategoricalCrossentropy()

def gradient(x, y, model):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_hat = model(x)
        loss = loss_object(y, y_hat)

    return tape.gradient(loss, model.trainable_weights)


if __name__ == '__main__':
    # Set the random seed for repeatability
    seed = 12227
    func.set_seeds(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='CustomModel1')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights_dir if args.weights_dir != '' else f'../weights/{dataset_name}/{architecture}_1'
    model_name = args.model_name
    epochs = args.epochs

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(model_name, flush=True)

    # Create directory for weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    X_train, y_train, X_test, y_test = func.cifar_resnet_data()
    #X_train, y_train = func.subsampling(X_train, y_train, p=0.01)  # Subsampling for testing
    
    # Load a model
    model = custom_models.build_model(model_name, input_shape=(32, 32, 3), num_classes=10)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')

    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)

    # Loop through epochs and load weights to compute gradient similarity
    for ep in range(1, epochs+1):
        # Load the saved weights for the current epoch
        weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{ep:02d}.weights.h5')
        if os.path.exists(weights_path):
            print(f"Loading weights for epoch {ep}...", flush=True)
            model.load_weights(weights_path)
        else:
            print(f"Weights for epoch {ep} not found. Skipping...", flush=True)
            continue

        # Compute gradients for the current epoch
        grad_batch = []
        for batch in gen_batches(X_train.shape[0], 128):
            grad = gradient(tf.convert_to_tensor(X_train[batch], tf.float32), tf.convert_to_tensor(y_train[batch]), model)
            tmp = grad[0].numpy().reshape(-1)
            for i in range(1, len(grad)):
                tmp = np.hstack((tmp, grad[i].numpy().reshape(-1)))
            grad_batch.append(tmp)

        # Compute cosine distance between gradient batches
        cosine = []
        for i in range(0, len(grad_batch)):
            for j in range(i + 1, len(grad_batch)):
                cosine.append(distance.cosine(grad_batch[i], grad_batch[j]))
        print(f"Epoch {ep}: Minimum cosine distance between gradients: {min(cosine)}")

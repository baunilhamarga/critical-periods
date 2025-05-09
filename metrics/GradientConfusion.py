import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from sklearn.utils import gen_batches
import os
import argparse
import datetime

import sys
sys.path.append('../')

from utils import custom_functions as func
from models import ResNetN
import pandas as pd

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
    parser.add_argument('--architecture', type=str, default='ResNet44')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--starting_epoch', type=int, default=0, help='Starting epoch for training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for gradient computation')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights_dir if args.weights_dir != '' else f'../weights/{dataset_name}/{architecture}_best_weights'
    model_name = args.model_name
    epochs = args.epochs
    starting_epoch = args.starting_epoch
    batch_size = args.batch_size

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(model_name, flush=True)

    # Create directory for weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    X_train, y_train, X_test, y_test = func.cifar_resnet_data()
    #X_train, y_train = func.subsampling(X_train, y_train, p=0.01)  # Subsampling for testing
    
    # Load a model
    model = ResNetN.build_model(model_name, input_shape=(32, 32, 3), num_classes=10, N_layers=44)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}.weights.h5')

    # Initialize the list to store all results
    all_results = []

    # Loop through epochs and load weights to compute gradient similarity
    for epoch in range(starting_epoch, epochs+1):
        print(f"Current time: {datetime.datetime.now()}", flush=True)
        # Load the saved weights for the current epoch
        weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5') if epoch > 0 else random_weights_path
        if os.path.exists(weights_path):
            print(f"Loading weights for epoch {epoch}...", flush=True)
            model.load_weights(weights_path)
        else:
            print(f"Weights for epoch {epoch} not found. Skipping...", flush=True)
            continue

        # Compute gradients for the current epoch
        grad_batch = []
        for batch in gen_batches(X_train.shape[0], batch_size=batch_size):
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

        results = {
            'epoch': epoch,
            'min_cosine_distance': min(cosine),
            'avg_cosine_distance': np.mean(cosine),
            'max_cosine_distance': max(cosine),
            'median_cosine_distance': np.median(cosine),
            'min_cosine_similarity': 1 - max(cosine),
            'avg_cosine_similarity': 1 - np.mean(cosine),
            'max_cosine_similarity': 1 - min(cosine),
            'median_cosine_similarity': 1 - np.median(cosine)
        }
        print(results)
        all_results.append(results)

    # Convert the results to a DataFrame
    df = pd.DataFrame(all_results)

    # Define the path to save the CSV file
    csv_path = f'../plots/gradient_confusion_results_{model_name}_{dataset_name}_batch_size_{batch_size}.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")

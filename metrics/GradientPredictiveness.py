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

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights_dir if args.weights_dir != '' else f'../weightsHeitor/{dataset_name}/{architecture}_best_weights'
    model_name = args.model_name
    epochs = args.epochs
    starting_epoch = args.starting_epoch


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
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')

    
    grad_epoch = []
    batch_size = 512

    # Loop through epochs and load weights to compute gradient of each layer in each epoch
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

        # Compute gradients for the current epoch for each batch
        grad_batch = []
        for batch in gen_batches(X_train.shape[0], batch_size):
            grad = gradient(tf.convert_to_tensor(X_train[batch], tf.float32), tf.convert_to_tensor(y_train[batch]), model)
            tmp = grad[0].numpy().reshape(-1)
            for i in range(1, len(grad)):
                tmp = np.hstack((tmp, grad[i].numpy().reshape(-1)))
            grad_batch.append(tmp)
        
        grad_epoch.append(grad_batch)


    # Compute cosine similarity between epoch t and t-1
    cossine_similarity = []
    for i in range(2, epochs+1):
        cosine_similarity_epoch = []
        # Iterate through each batch
        for j in range(0, len(grad_epoch[i-1])):
            cosine_similarity_epoch.append(1 - distance.cosine(grad_epoch[i][j], grad_epoch[i-1][j]))
        cosine_similarity.append(cosine_similarity_epoch)

    # Get mean of each element of cossine_similarity
    cossine_similarity = cossine_similarity.mean(axis=0)

    maximum = max(cosine_similarity)
    results = {
        'min_cosine_similarity': min(cosine_similarity),
        'avg_cosine_similarity': np.mean(cosine_similarity),
        'max_cosine_similarity': max(cosine_similarity),
        'median_cosine_similarity': np.median(cosine_similarity)
    }
    print(results)

    # Convert the results to a DataFrame
    epochs = list(range(1, len(cosine_similarity) + 1))

    # Create a DataFrame with the cosine similarity values and the corresponding epochs
    df = pd.DataFrame({
        'epoch': epochs,
        'gradient_predictiveness': cosine_similarity
    })


    # Define the path to save the CSV file
    csv_path = f'../plots/{model_name}_{dataset_name}_{batch_size}_gradient_predictiveness_results.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")

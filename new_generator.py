import tensorflow as tf
import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.spatial.distance import cosine
import os
import argparse
from models import ResNetN
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50

def infinite_generator(generator):
    """Wrap a generator so that it yields batches indefinitely."""
    while True:
        for batch in generator:
            yield batch

def generate_data_augmentation(X_train, seed=12227):
    print('Using real-time data augmentation.')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train, seed=seed)
    return datagen

# Global variables for storing initial weights and cosine distances
initial_weights = None
cosine_distances_mean = []
cosine_distances_concat = []

def save_initial_weights(model, layer_names):
    global initial_weights
    initial_weights = [model.get_layer(name).get_weights()[0] for name in layer_names]

def calculate_cosine_distance(weights1, weights2):
    distances = []
    distance = np.array([cosine(w1.flatten(), w2.flatten()) for w1, w2 in zip(weights1, weights2)])
    distances.append(distance)
    return distances

def check_rotation_angle_criterion(model, layer_names, window_size=5, epoch=6, epochs=200):
    global initial_weights, cosine_distances_mean, cosine_distances_concat

    # Get current weights from the specified layers
    current_weights = [model.get_layer(name).get_weights()[0] for name in layer_names]
    # Calculate mean cosine distance between current and initial weights
    cosine_distance_mean = np.mean(calculate_cosine_distance(initial_weights, current_weights))
    cosine_distances_mean.append(cosine_distance_mean)
    print(f"\nMean cosine distance: {cosine_distance_mean:.4f}", flush=True)

    # If we have enough points, check the regression slope
    if len(cosine_distances_mean) >= window_size:
        recent_distances = np.array(cosine_distances_mean[-window_size:]).reshape(-1, 1)
        epochs_ndarray = np.arange(1, epochs+1).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Normalizing epochs for regression (optional)
        recent_distances_norm = recent_distances
        epochs_norm = scaler.fit_transform(epochs_ndarray)
        recent_distances_norm_1d = recent_distances_norm.flatten()
        epochs_norm_1d = epochs_norm.flatten()
        epochs_window = epochs_norm_1d[epoch-window_size:epoch]
        print(epochs_window)
        print(recent_distances_norm_1d)
        
        slope, intercept = np.polyfit(epochs_window, recent_distances_norm_1d, 1)
        angle = np.degrees(np.arctan(slope))
        print(f"\nAngle: {angle:.2f}°", flush=True)
        
        if angle < 45:
            return True

    return False

def get_kernel_layer_names(model):
    layer_names = []
    for l in model.layers:
        if len(l.weights) > 0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet56')
    parser.add_argument('--dataset', type=str, default='imagenet5')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--baseline', type=str, default='False', help='True = All epochs k = 3, False = Criterion will be implemented')
    args = parser.parse_args()

    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    verbose = args.verbose
    criterion_met = False if args.baseline == 'False' else True

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)
    os.makedirs(weights_dir, exist_ok=True)

    if 'tiny' in dataset_name.lower():
        num_classes = 200
        data_path = f'/home/vm03/Datasets/tiny_imagenet_train.npz'
    elif 'imagenet' in dataset_name.lower():
        match = re.search(r'(\d+)', dataset_name)
        if match:
            num_classes = int(match.group(1))
            data_path = f'/home/vm03/Datasets/imagenet{num_classes}_cls.npz'
        else:
            raise ValueError("Unsupported dataset. Dataset name should contain the number of classes, e.g., CIFAR10 or ImageNet30.")
    elif 'eurosat' in dataset_name.lower():
        num_classes = 10
        data_path = f'/home/vm03/Datasets/eurosat.npz'

    data = np.load(data_path)
    if 'X_val' in data:
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_val'], data['y_val']
    else:
        X_train, y_train = data['X_train'], data['y_train']
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    
    y_train = np.eye(num_classes)[y_train.reshape(-1)]
    y_test = np.eye(num_classes)[y_test.reshape(-1)]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    num_classes = y_train.shape[1]

    match = re.search(r'ResNet(\d+)', architecture)
    if match:
        N_layers = int(match.group(1))
        print(f"\nNumber of Layers: {N_layers}", flush=True)
    else:
        raise ValueError("Arquitetura não suportada. Suporta apenas formatos como ResNetXX, onde XX é o número de camadas.")

    if model_name.lower() == 'resnet50':
        model = ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=X_train[0].shape,
            pooling=None,
            classes=num_classes,
            classifier_activation="softmax"
        )
    else:
        model = ResNetN.build_model(model_name, input_shape=X_train[0].shape, num_classes=num_classes, N_layers=N_layers)

    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}.weights.h5')
    func.load_or_create_weights(model, random_weights_path)

    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Configure data augmentation using your custom generator
    datagen = func.generate_data_augmentation(X_train)
    lr_scheduler_callback = LearningRateScheduler(func.scheduler)
    callbacks = [lr_scheduler_callback]

    epochs = 200
    batch_size = 32

    # Simulate k-times repetition by increasing the steps per epoch.
    k = 3  # augmentation factor
    num_samples_epoch = k * X_train.shape[0]
    steps_per_epoch = math.ceil(num_samples_epoch / batch_size)
    print(f"{num_samples_epoch} samples per epoch (k={k}), grouped into batches of {batch_size}.", flush=True)

    # Wrap the generator so that it repeats indefinitely.
    dataflow = infinite_generator(datagen.flow(X_train, y_train, batch_size=batch_size, seed=seed, shuffle=True))
    
    # Get kernel layer names for later use.
    layer_names = get_kernel_layer_names(model)
    if initial_weights is None:
        save_initial_weights(model, layer_names)
        
    epochs_annealing = []
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        model.fit(
            dataflow,
            steps_per_epoch=steps_per_epoch,
            epochs=epoch, initial_epoch=epoch - 1,
            verbose=verbose, callbacks=callbacks,
            validation_data=(X_test, y_test),
            validation_freq=5
        )
        
        if (epoch in epochs_annealing) or epoch % 10 == 0:
            weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5')
            print(f"\nSaving model weights. Epoch {epoch}.")
            model.save_weights(weights_path)

        if not criterion_met:
            if check_rotation_angle_criterion(model, layer_names, window_size=5, epoch=epoch, epochs=epochs):
                print(f"Criterion met at epoch {epoch}. Adjusting k value.")
                # Change k to 1 and reconfigure steps_per_epoch accordingly.
                k = 1
                num_samples_epoch = k * X_train.shape[0]
                steps_per_epoch = math.ceil(num_samples_epoch / batch_size)
                dataflow = infinite_generator(datagen.flow(X_train, y_train, batch_size=batch_size, seed=seed, shuffle=True))

                cosine_distances_mean = []
                criterion_met = True

                weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5')
                print(f"\nSaving model weights. Epoch {epoch}.")
                model.save_weights(weights_path)

                epochs_diff = epochs - epoch
                annealing_ratio = [0.99, 0.95, 0.90, 0.85, 0.80, 0.50, 0.875]
                for ratio in annealing_ratio:  
                    annealing_epoch = epoch + math.ceil(epochs_diff * ratio)
                    epochs_annealing.append(annealing_epoch)
                    print(f"\nAnnealing ratio: {ratio}. Annealing epoch: {annealing_epoch}")

    # Evaluate the model after training.
    y_pred = model.predict(X_test, verbose=0)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')
    
    # Save after training
    weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5')
    print(f"\nSaving model weights. Epoch {epochs}.")
    model.save_weights(weights_path)

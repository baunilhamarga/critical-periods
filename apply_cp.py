import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse
from models import ResNetN
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb
import pandas as pd
import re



if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet44')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    verbose = args.verbose

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Get the number of classes from the dataset name
    match = re.search(r'([a-zA-Z]+)(\d+)', dataset_name)
    if match:
        num_classes = int(match.group(2))
    else:
        raise ValueError("Unsupported dataset. Dataset name should contain the number of classes, e.g., CIFAR10 or ImageNet30.")

    # Load the dataset
    if match.group(1).lower() == "cifar":
        X_train, y_train, X_test, y_test = func.cifar_resnet_data(cifar_type=num_classes)
    
    elif match.group(1).lower() == "imagenet":
        data = np.load(f'/home/vm03/Datasets/imagenet{num_classes}_cls.npz')
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_val'], data['y_val']
        
    else:
        raise ValueError("Unsupported dataset. Supported datasets are CIFAR and ImageNet.")
    
    # Convert y_train from int to one-hot encoding
    if len(y_train.shape) == 1:
        y_train = np.eye(num_classes)[y_train.reshape(-1)]
        y_test = np.eye(num_classes)[y_test.reshape(-1)]

    # Load a model
    if model_name.lower().startswith('resnet'):
        N_layers = int(model_name.lower().split('resnet')[-1])
        input_shape = X_train.shape[1:]
        model = ResNetN.build_model(model_name, input_shape=input_shape, num_classes=num_classes, N_layers=N_layers)
    else:
        raise ValueError(f'Invalid model name: {model_name}.')

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}.weights.h5')
    
    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)
    
    # Configure the optimizer
    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Configure Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(func.scheduler)

    # Set up the ModelCheckpoint callback to save the model every n epochs
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{{epoch:02d}}.weights.h5"),
        save_weights_only=True,  # Save only the weights
        save_freq='epoch',  # Save at the end of every epoch
        verbose=1
    )
    
    gradient_confusion = cb.GradientConfusion(X_train, y_train, initial_distribution_size=10, batch_size=32)
    
    callbacks = [lr_scheduler_callback, gradient_confusion]

    # Set manual epoch loop configuration
    epochs = 200
    batch_size = 32
    critical_window = 100  # Number of epochs to look for the end of the critical period

    # Repeat the data k times, datagen will transform
    k = 1
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    # Train with augmentation until the critical period metric stops it
    print(f"{X_aug.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)
    history_aug = model.fit(
        datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
        epochs=critical_window,
        verbose=verbose, callbacks=callbacks,
        validation_data = (X_test, y_test),
        validation_freq=5
    )
    func.save_history_to_csv(history_dict=history_aug.history, csv_path=f'history_{model_name}_{dataset_name}_aug.csv')
    
    # Get the total number of epochs that were run
    stopped_epoch = len(history_aug.history['loss'])
    
    if stopped_epoch < epochs:
        # Resume training with reduced augmentation until the end
        print(f"{X_train.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)
        history_no_aug = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size, seed=seed, shuffle=True),
            epochs=epochs, initial_epoch=stopped_epoch,
            verbose=verbose, callbacks=callbacks[:-1],
            validation_data = (X_test, y_test),
            validation_freq=5
        )
        func.save_history_to_csv(history_dict=history_no_aug.history, csv_path=f'history_{model_name}_{dataset_name}_no_aug.csv', starting_epoch=stopped_epoch+1)
    
    # Evaluate model after training
    y_pred = model.predict(X_test, verbose=0)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

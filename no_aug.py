import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse
from models import ResNetN
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb
import datetime
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet44')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--k', type=int, default=1, help='Number of times to repeat the data')
    parser.add_argument('--output_path', type=str, default='', help='Path to save the output files')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--starting_epoch', type=int, default=1, help='Starting epoch for training')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    k = args.k
    verbose = args.verbose
    output_path = args.output_path if args.output_path != '' else weights_dir
    starting_epoch = args.starting_epoch

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load the CIFAR-10 dataset
    cifar_type = int(dataset_name.lower().split('cifar')[-1])
    X_train, y_train, X_test, y_test = func.cifar_resnet_data(cifar_type=cifar_type)

    # Load a model
    N_layers = int(architecture.lower().split('resnet')[-1])
    num_classes = y_train.shape[1]
    model = ResNetN.build_model(model_name, input_shape=(32, 32, 3), num_classes=num_classes, N_layers=N_layers)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}.weights.h5')
    
    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)
    
    # Configure Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(func.scheduler)
    
    callbacks = [lr_scheduler_callback]

    # Repeat the data k times, datagen will transform
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    print(f"{X_aug.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)
    
    accuracies = []
    starting_epoch = starting_epoch
    
    # Configure the optimizer
    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Train with no augmentation loop
    for initial_epoch in range(starting_epoch, n_epochs+1):
        print(f"Current time: {datetime.datetime.now()}", flush=True)
        
        log = pd.read_csv('./logs/no_aug_log.csv')
        
        while (initial_epoch - 1, model_name, dataset_name, k) in zip(log['initial_epoch'], log['model'], log['dataset'], log['k']):
            initial_epoch += 1
        
        if initial_epoch > n_epochs:
            print(f"no_aug job complete for {model_name} on {dataset_name}.", flush=True)
            break
            
        log_entry = {
            'model': model_name,
            'dataset': dataset_name,
            'k': k,
            'initial_epoch': initial_epoch - 1,
            'accuracy': None
        }
        
        func.log_to_csv(log_entry, log_file_name='no_aug_log.csv')
        
        # Load the epoch weights with augmentation
        if initial_epoch != 1:
            print(f"\nLoading weights already trained for {initial_epoch-1} epochs with data augmentation.", flush=True)
            model.load_weights(os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{initial_epoch-1:02d}.weights.h5"))
        
        print(f"Training from the start of epoch {initial_epoch} to the end of epoch {n_epochs} with k={k}.", flush=True)
        
        model.fit(
            datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
            epochs=n_epochs, initial_epoch=initial_epoch - 1,
            verbose=verbose, callbacks=callbacks,
            validation_data = (X_test, y_test),
            validation_freq=5
        )

        # Evaluate model after training
        y_pred = model.predict(X_test, verbose=0)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print(f'Final accuracy trained from the start of epoch {initial_epoch}: {accuracy:.4f}')
        
        accuracies.append((initial_epoch - 1, accuracy))
        
        log_entry = {
            'model': model_name,
            'dataset': dataset_name,
            'k': k,
            'initial_epoch': initial_epoch - 1,
            'accuracy': accuracy
        }
        
        func.log_to_csv(log_entry, log_file_name='no_aug_log.csv', delete_duplicate=True)
        
        # Save the model weights at the end of training
        print(f"Saving weights trained with k={k} from epoch {initial_epoch} to {n_epochs}.", flush=True)
        model.save_weights(os.path.join(weights_dir, f"{model_name}_{dataset_name}_no_aug_from_epoch_{initial_epoch:02d}.weights.h5"), overwrite=True)
        
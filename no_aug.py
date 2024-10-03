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

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet44')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--output_path', type=str, default='', help='Path to save the output files')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--starting_epoch', type=int, default=1, help='Starting epoch for training')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}_best_weights'
    model_name = args.model_name
    verbose = args.verbose
    output_path = args.output_path if args.output_path != '' else weights_dir
    starting_epoch = args.starting_epoch

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load the CIFAR-10 dataset
    X_train, y_train, X_test, y_test = func.cifar_resnet_data()

    # Load a model
    model = ResNetN.build_model(model_name, input_shape=(32, 32, 3), num_classes=10, N_layers=44)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')
    
    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)
    
    # Configure the optimizer
    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Configure Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(func.scheduler)
    
    callbacks = [lr_scheduler_callback]

    # Set manual epoch loop configuration
    epochs = 200
    batch_size = 16

    # Repeat the data k times, datagen will transform
    k = 1
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    print(f"{X_aug.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)
    
    accuracies = []
    starting_epoch = starting_epoch
    
    # Train with no augmentation loop
    for initial_epoch in range(starting_epoch, epochs+1):
        print(f"Current time: {datetime.datetime.now()}", flush=True)
        # Load the epoch weights with augmentation
        if initial_epoch != 1:
            print(f"\nLoading weights already trained for {initial_epoch-1} epochs with data augmentation.", flush=True)
            model.load_weights(os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{initial_epoch-1:02d}.weights.h5"))
        
        print(f"Training from the start of epoch {initial_epoch} to the end of epoch {epochs} with k={k}.", flush=True)
        
        model.fit(
            datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
            epochs=epochs, initial_epoch=initial_epoch - 1,
            verbose=verbose, callbacks=callbacks,
            validation_data = (X_test, y_test),
            validation_freq=5
        )

        # Evaluate model after training
        y_pred = model.predict(X_test, verbose=0)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print(f'Final accuracy trained from the start of epoch {initial_epoch}: {accuracy:.4f}')
        
        accuracies.append((initial_epoch - 1, accuracy))
        
        print(f"[(initial_epoch , accuracies)] = {accuracies}", flush=True)
        
        # Save the model weights at the end of training
        print(f"Saving weights trained with k={k} from epoch {initial_epoch} to {epochs}.", flush=True)
        model.save_weights(os.path.join(weights_dir, f"{model_name}_{dataset_name}_no_aug_from_epoch_{initial_epoch:02d}.weights.h5"), overwrite=True)
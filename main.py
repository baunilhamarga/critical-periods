import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse
from models import ResNetN
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet32', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use for training and evaluation')
    parser.add_argument('--weights', type=str, default='', help='Directory to save or load model weights')
    parser.add_argument('--model_name', type=str, default='', help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--k', type=int, default=3, help='Number of times to repeat the data')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    verbose = args.verbose
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    k = args.k

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
    model = ResNetN.build_model(model_name, input_shape=X_train[0].shape, num_classes=num_classes, N_layers=N_layers)

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
    
    callbacks = [lr_scheduler_callback, checkpoint_callback]

    # Repeat the data k times, datagen will transform
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    print(f"{X_aug.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)

    model.fit(
        datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
        epochs=n_epochs,
        verbose=verbose, callbacks=callbacks,
        validation_data = (X_test, y_test),
        validation_freq=5
    )

    # Evaluate model after training
    y_pred = model.predict(X_test, verbose=0)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

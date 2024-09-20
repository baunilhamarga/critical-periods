import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse

# Set seeds for repeatability
def set_seeds(seed=12227):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Load CIFAR-10 dataset
def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize the data to [0, 1] range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 25:
        return lr
    elif epoch < 75:
        return 1e-3
    else:
        return 1e-4

# Save random initialized weights
def save_random_weights(model, path):
    model.save_weights(path)

# Check if random weights exist, else create and save them
def load_or_create_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"{weights_path} not found. Creating random starting weights.")
        save_random_weights(model, weights_path)
    else:
        print(f"Loading random starting weights from {weights_path}.")
    model.load_weights(weights_path)

if __name__ == '__main__':
    seed = 12227
    set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(model_name, flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load the CIFAR-10 dataset
    X_train, y_train, X_test, y_test = load_cifar10_data()
    
    train_size = len(X_train)
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    # Load a ResNet model (without the top layer, and with input shape for CIFAR-10)
    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'random_starting_weights_{model_name}_.weights.h5')
    
    # Load or create the random starting weights using the seed
    load_or_create_weights(model, random_weights_path)
    
    # Configure the optimizer
    lr = 0.01
    optimizer = SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Configurr Data Augmentation
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
    datagen.fit(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

    # Set up the ModelCheckpoint callback to save the model every n epochs
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{{epoch:02d}}.weights.h5"),
        save_weights_only=True,  # Save only the weights
        save_freq='epoch',  # Save at the end of every epoch
        verbose=1
    )
    
    callbacks = [lr_scheduler_callback, checkpoint_callback]

    # Set manual epoch loop
    epochs = 200
    batch_size = 128
    steps_per_epoch = len(X_train) // batch_size

    # Epoch loop
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                  steps_per_epoch=steps_per_epoch,
                  epochs=epoch, initial_epoch=epoch - 1,
                  verbose='auto', callbacks=callbacks)

        if epoch % 5 == 0:
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)

    # Evaluate model after training
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

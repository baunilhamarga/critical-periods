import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse

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

if __name__ == '__main__':
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(model_name, flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights, exist_ok=True)

    # Load the CIFAR-10 dataset
    X_train, y_train, X_test, y_test = load_cifar10_data()
    #X_train, y_train = X_train[:258], y_train[:258]

    # Load a ResNet model (without the top layer, and with input shape for CIFAR-10)
    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

    # Configure the optimizer
    lr = 0.01
    optimizer = SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Configure the Data Augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    datagen.fit(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Set up the ModelCheckpoint callback to save the model every n epochs
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(weights, f"{model_name}_{dataset_name}_epoch_{{epoch:02d}}.weights.h5"),
        save_weights_only=True,  # Save only the weights
        save_freq='epoch',  # Save at the end of every epoch
        verbose=1
    )
    
    epochs = 10
    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=128),
              steps_per_epoch=len(X_train) // 128 + 2,
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[lr_scheduler_callback, checkpoint_callback])

    # Save the model weights after training
    model.save_weights(os.path.join(weights, f'{model_name}_{dataset_name}_{epochs}.weights.h5'))

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

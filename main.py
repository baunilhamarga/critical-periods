import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import argparse
from models import custom_models
from utils import custom_functions as func
from utils import custom_callbacks as callbacks

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='CustomModel1')
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
    X_train, y_train, X_test, y_test = func.cifar_resnet_data()
    
    train_size = len(X_train)
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    # Load a model
    model = custom_models.build_model(model_name, input_shape=(32, 32, 3), num_classes=10)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')
    
    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)
    
    # Configure the optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Configurr Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(func.lr_scheduler)

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
    verbose = 2 # 0 = silent, 1 = progress bar, 2 = one line per epoch

    # Epoch loop
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                  steps_per_epoch=steps_per_epoch,
                  epochs=epoch, initial_epoch=epoch - 1,
                  verbose=verbose, callbacks=callbacks, validation_data = (X_test, y_test))

        if epoch % 5 == 0:
            acc = model.evaluate(X_test, y_test)[1]
            print(f'Test Accuracy of Model: {acc}', flush=True)

    # Evaluate model after training
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

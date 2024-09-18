import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from keras import backend
from sklearn.metrics.classification import accuracy_score
import argparse
import sys

sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks

if __name__ == '__main__':
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='../../architectures/CIFAR10/ResNet20')
    parser.add_argument('--weights', type=str, default='../../weights/CIFAR10/ResNet20++')
    parser.add_argument('--model_name', type=str, default='')

    args = parser.parse_args()
    architecture = args.architecture
    weights = args.weights
    model_name = args.model_name

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(model_name, flush=True)

    model = func.load_model(architecture,
                            weights)

    X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=False)

    # Configura o Data Augmentation.
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
        vertical_flip=False)  # randomly flip images
    # preprocessing_function=func.cutout)
    datagen.fit(X_train)

    lr = 0.01
    schedule = [(25, 1e-3), (75, 1e-4)]

    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    model = func.optimizer_compile(model, 'VGG16')

    for epoch in range(1, 100):

        model.fit_generator(datagen.flow(X_train, y_train),
                            steps_per_epoch=5000,
                            epochs=epoch, initial_epoch=epoch - 1,
                            verbose=2, callbacks=callbacks)
        if epoch % 20 == 0:
            model.save_weights('{}_epoch{}.h5'.format(model_name, epoch))

        if epoch % 5 == 0:
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)

    model.save_weights('{}.h5'.format(model_name))

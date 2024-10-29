import numpy as np
import keras
from keras.callbacks import Callback
import pandas as pd
import custom_functions as func

class LearningRateScheduler(Callback):

    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        lr = self.init_lr
        for i in range(0, len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print('Learning rate:{}'.format(lr))
        #K.set_value(self.model.optimizer.lr, lr)
        keras.backend.set_value(self.model.optimizer.lr, lr)

class SavelModelScheduler(Callback):

    def __init__(self, file_name='', schedule=[1, 25, 50, 75 , 100, 150]):
        super(Callback, self).__init__()
        self.schedule = schedule
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs={}):
        if epoch in self.schedule:
            model_name = self.file_name+'epoch{}'.format(epoch)
            print('Epoch %05d: saving model to %s' % (epoch, model_name))
            self.model.save_weights(model_name, overwrite=True)
            with open(model_name + '.json', 'w') as f:
                f.write(self.model.to_json())

def custom_stopping(value=0.5, verbose=0):
    early = keras.callbacks.EarlyStoppingByLossVal(monitor='val_loss', value=value, verbose=verbose)
    return early

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value=0.95, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        # warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class GradientConfusion(Callback):
    def __init__(self, X_train, y_train, metric_frequency=1, initial_distribution_size=10, threshold=1.5):
        """
        Gradient Confusion Critical Period Identification.

        Args:
            metric_frequency (int): Epoch frequency for max cosine similarity calculation.
            check_interval (int): Number of epochs after which to check for outliers.
            threshold (float): Threshold multiplier for outlier detection.
        """
        super(GradientConfusion, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.frequency = metric_frequency
        self.check_interval = initial_distribution_size
        self.threshold = threshold
        self.max_cosine_similarity_values = []

    def on_epoch_end(self, epoch, logs=None):
        # Get the new max cosine similarity value for the current epoch
        if (epoch + 1) % self.frequency == 0:
            new_max_similarity = func.calculate_max_cosine_similarity(self.X_train, self.y_train, self.model)
        
            # Log metric to logs dictionary
            logs['max_cosine_similarity'] = new_max_similarity

            # Check for outlier condition based on check interval
            if (epoch + 1) % self.check_interval == 0 and len(self.max_cosine_similarity_values) > 1:
                # Call the outlier detection function
                if func.is_outlier(self.max_cosine_similarity_values[:-1], new_max_similarity, threshold=self.threshold, mode='low'):
                    print(f"Stopping training at epoch {epoch + 1} due to detected low outlier in max cosine similarity.")
                    self.model.stop_training = True
                    
            # Append point to the history
            self.max_cosine_similarity_values.append(new_max_similarity)

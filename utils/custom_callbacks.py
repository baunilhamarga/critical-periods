import numpy as np
import keras
from keras.callbacks import Callback

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


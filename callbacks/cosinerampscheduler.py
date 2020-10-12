import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import math


class CosineRampLearningRateScheduler(Callback):
    def __init__(self, total_number_of_batches, ramp_up_fraction, ramp_down_fraction, learning_rate):
        """
        Consine Ramp Learning rate scheduler.
        At the beginning of every batch, this callback sets the learning rate of the optimizer
        based on the following parameters.
        :param total_number_of_batches: Total number of training batches summed across all the epochs
        :param ramp_up_fraction: Fraction of the training batches used for ramping up the learning rate at the beginning
        :param ramp_down_fraction: Fraction of the training batches used for ramping down the learning rate at the end
        :param learning_rate: Base learning rate
        """
        super(CosineRampLearningRateScheduler, self).__init__()
        self.total_number_of_batches = total_number_of_batches
        self.ramp_up_fraction = ramp_up_fraction
        self.ramp_down_fraction = ramp_down_fraction
        self.learning_rate = learning_rate
        self.steps = 0
        self.ramp_up_end_iter = self.total_number_of_batches * self.ramp_up_fraction
        self.ramp_down_start_iter = self.total_number_of_batches * (1 - self.ramp_down_fraction)

    def on_train_begin(self, logs=None):
        self.steps = 0

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.compute_ramped_lrate(self.steps)
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        self.steps += 1

    def on_train_batch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)

    def compute_ramped_lrate(self, i):
        if (self.ramp_up_fraction > 0.0) and (i <= self.ramp_up_end_iter):
            t = i / (self.ramp_up_fraction * self.total_number_of_batches)
            return self.learning_rate * (0.5 - math.cos(t * math.pi) / 2)

        elif (self.ramp_down_fraction > 0.0) and (i >= self.ramp_down_start_iter):
            t = (i - self.ramp_down_start_iter) / (self.ramp_down_fraction * self.total_number_of_batches)
            return self.learning_rate * (0.5 + math.cos(t * math.pi) / 2) ** 2

        else:
            return self.learning_rate

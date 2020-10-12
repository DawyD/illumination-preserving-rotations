from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np


class TestAndLogCallback(Callback):
    def __init__(self, validation_sets, validation_set_names=None, verbose=0,
                 log_dir="logs", scalar_freq=100, image_freq=20, channels=None, epoch_multiplier=1):
        """
        :param validation_sets: List of data generators
        :param validation_set_names: List of validation set names of the same length as 'validation_sets'
        :param verbose: verbosity mode, 1 or 0
        :param log_dir: Directory for TensorBoard log
        :param scalar_freq: Frequency of saving the scalar metrics in number of epochs
        :param image_freq: Frequency of saving the images in number of epochs]
        :param channels: Specifies the number of channels used for the image summary. If None, all channels are used.
        :param epoch_multiplier: Specifies the number of train data iterations in a single epoch
        """
        super(TestAndLogCallback, self).__init__()
        self.validation_sets = validation_sets
        if validation_set_names is None:
            self.validation_set_names = ["valid{:d}".format(i) for i in range(len(validation_sets))]
        else:
            self.validation_set_names = validation_set_names
        self.verbose = verbose

        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.summary_writer.set_as_default()

        self.scalar_freq = scalar_freq
        self.image_freq = image_freq
        self.channels = channels
        self.epoch_multiplier = epoch_multiplier

    def on_train_end(self, logs=None):
        self.summary_writer.close()

    def on_epoch_end(self, epoch, logs=None):
        # Log scalars
        if (epoch * self.epoch_multiplier) % self.scalar_freq == 0:
            with self.summary_writer.as_default():
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                tf.summary.scalar("train/learning_rate", lr, step=epoch * self.epoch_multiplier)

            for validation_set, validation_set_name in zip(self.validation_sets, self.validation_set_names):
                if validation_set is None:
                    continue
                results = self.model.evaluate(validation_set, verbose=self.verbose)
                for metric, result in zip(self.model.metrics_names, results):
                    valuename = validation_set_name + '/' + metric
                    with self.summary_writer.as_default():
                        tf.summary.scalar(valuename, result, step=epoch * self.epoch_multiplier)

        # Log images
        if (epoch * self.epoch_multiplier) % self.image_freq == 0:
            for validation_set, validation_set_name in zip(self.validation_sets, self.validation_set_names):
                if validation_set is None:
                    continue

                batch_x, batch_y, batch_w = validation_set[0]
                predictions = self.model(batch_x, training=False)

                imsum = self.create_defect_image_summary(batch_x, batch_y, predictions, channels=self.channels)

                with self.summary_writer.as_default():
                    tf.summary.image(validation_set_name + '/predictions', imsum, step=epoch * self.epoch_multiplier)

    @staticmethod
    def create_defect_image_summary(images, gt, predictions, max_nr=3, channels=None):
        """
        Combines the images, predictions, and masks to form a image summary
        :param images: Batch of input images
        :param gt: Batch of ground truth segmentation masks
        :param predictions: Batch of predicted defect segmentation masks
        :param max_nr: Maximal number of summary images from this batch
        :param channels: Specifies the number of channels used for the image summary. If None, all channels are used.
        :returns Summary image
        """
        images = images[:max_nr]
        if channels is not None:
            images = images[..., :channels]
        predictions = predictions[:max_nr]
        masks = gt[:max_nr]

        if images.shape[-1] == 1:
            images = np.tile(images, (1, 1, 1, 3))
        if images.shape[-1] == 2:
            images = np.concatenate((images, images[..., 0]), axis=-1)
        if images.shape[-1] > 3:
            images = np.concatenate((np.std(images, axis=-1, keepdims=True),
                                     np.std(images, axis=-1, keepdims=True),
                                     np.mean(images, axis=-1, keepdims=True)), axis=-1)

        images -= np.amin(images, axis=(1, 2), keepdims=True)
        images /= np.amax(images, axis=(1, 2), keepdims=True)
        masks = np.tile(np.cast[np.float32](masks), (1, 1, 1, 3))
        predictions = np.tile(predictions, (1, 1, 1, 3))
        out_comb = np.concatenate((images, masks, predictions), axis=2)

        return np.clip(out_comb, 0, 1)

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math as m
from time import time
from datetime import datetime
from os import path, makedirs
from PIL import Image
import logging
from tqdm import tqdm

from models.net import Net

from data.dataloader import DataIterator


class Metrics:
    def __init__(self, prefix):
        """
        Groups several keras metric objects: CrossEntropy, BinaryAccuracy, Precision, Recall, AuROC, AuPRC
        :param prefix: prefix for the names of the keras metrics, which will be also used in summary file
                       (e.g 'train_to_test' or 'test')
        """
        self.prefix = prefix
        self.loss = tf.keras.metrics.Mean(name=prefix + '_loss')
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name=prefix + '_accuracy')
        self.precision = tf.keras.metrics.Precision(name=prefix + '_precision')
        self.recall = tf.keras.metrics.Recall(name=prefix + '_recall')
        self.auroc = tf.keras.metrics.AUC(name=prefix + '_AUROC')
        self.auprc = tf.keras.metrics.AUC(curve="PR", name=prefix + '_AUPRC')

    def update(self, loss, predictions, masks):
        """
        Updates the state individual metrics by a batch of predictions.
        Note that metrics working with binary predictions use threshold of 0.5
        :param loss: cross-entropy loss for the batch
        :param predictions: Predicted masks from the model
        :param masks: Ground truth segmentation masks
        """
        self.loss(loss)
        self.accuracy(masks, predictions)
        self.precision(masks, predictions)
        self.recall(masks, predictions)
        self.auroc(masks, predictions)
        self.auprc(masks, predictions)

    def update_t(self, loss, predictions, masks, t_predictions):
        """
        Updates the state individual metrics by a batch of predictions
        :param loss: cross-entropy loss for the batch
        :param predictions: Predicted masks from the model
        :param masks: Ground truth segmentation masks
        :param t_predictions: Thresholded masks that are used for updating Accuracy, Precision, Recall, and IoU
        """
        self.loss(loss)
        self.accuracy(masks, t_predictions)
        self.precision(masks, t_predictions)
        self.recall(masks, t_predictions)
        self.auroc(masks, predictions)
        self.auprc(masks, predictions)

    def reset(self):
        """
        Resets the states of all metrics
        """
        self.loss.reset_states()
        self.accuracy.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.auroc.reset_states()
        self.auprc.reset_states()

    def write(self, step):
        """
        Write the metrics to the default summary file
        """
        tf.summary.scalar(self.prefix + '/loss', self.loss.result(), step=step)
        tf.summary.scalar(self.prefix + '/accuracy', self.accuracy.result(), step=step)
        tf.summary.scalar(self.prefix + '/precision', self.precision.result(), step=step)
        tf.summary.scalar(self.prefix + '/recall', self.recall.result(), step=step)
        tf.summary.scalar(self.prefix + '/AuROC', self.auroc.result(), step=step)
        tf.summary.scalar(self.prefix + '/AuPRC', self.auprc.result(), step=step)


class Trainer:
    def __init__(self, detector: Net, rotate_inputs=False, rotate_normals=False, small_rotate_inputs=False,
                 opt_kwargs=None):
        """
        Trainer object for image segmentation tasks.
        :param detector: Image segmentation network
        :param rotate_inputs: True if input arrays should be randomly flipped and rotated by [0-360] degrees during training
        :param rotate_normals: True if while rotating input arrays the channels 1 (x) and 2 (y) should be rotated accordingly
        :param small_rotate_inputs: True if input arrays should be randomly rotated by [-4,4] degrees during training.
        :param opt_kwargs: dictionary containing optimizer args {optimizer: 'adam' or 'SGD', learning_rate:0.001, ... }
        """

        tf.keras.backend.clear_session()
        tf.keras.backend.set_image_data_format("channels_last")

        self.rotate_inputs = rotate_inputs
        self.rotate_normals = rotate_normals
        self.small_rotate_inputs = small_rotate_inputs

        self.detector = detector
        self.model = detector.net()

        self.loss_object = tf.keras.losses.BinaryCrossentropy()

        if opt_kwargs is None:
            opt_kwargs = {}
        optimizer = None
        if "optimizer" in opt_kwargs:
            optimizer = opt_kwargs["optimizer"]
            opt_kwargs.pop("optimizer")
        if optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(**opt_kwargs)
        elif optimizer is None or optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(**opt_kwargs)
        else:
            raise ValueError("Unknown optimizer")
        self.opt_kwargs = opt_kwargs

        self.metrics = {
            'test': Metrics("test"),
            'train_to_test': Metrics("train_to_test"),
        }

    @tf.function
    def train_step(self, images, segmentations, masks, lr):
        """
        One step of optimization of detector network given the batch of images, foreground segmentations, and masks
        :param images: Batch of input images
        :param segmentations: Batch of foreground segmentation masks
        :param masks: Batch of defect segmentation masks
        :param lr: Learning rate used in this step
        """
        segmentations = tf.cast(segmentations, tf.float32)
        masks = tf.cast(masks, tf.float32)

        if self.rotate_inputs:
            logging.info("Adding random image rotations")
            angles_rad = tf.random.uniform((images.shape[0],)) * 2 * m.pi  # generate random angles for each sample
            images = tfa.image.rotate(images, angles_rad, interpolation='BILINEAR', name="rotate_images")
            segmentations = tfa.image.rotate(segmentations, angles_rad, interpolation='NEAREST', name="rotate_segmentation")
            masks = tfa.image.rotate(masks, angles_rad, interpolation='NEAREST', name="rotate_masks")

            if self.rotate_normals:  # Rotate normal directions (x,y) assuming order of channels: [x, y, z]
                rot_matrices = tf.expand_dims(
                    tf.expand_dims(
                        tf.transpose(
                            tf.stack([(tf.cos(-angles_rad), -tf.sin(-angles_rad)),
                                      (tf.sin(-angles_rad), tf.cos(-angles_rad))], axis=0),
                            (1, 0, 2)),
                        axis=1),
                    axis=1)  # [batch_size, 1, 1, x, y]
                normals_xy = tf.expand_dims(images[..., 0:2], axis=4)  # [batch_size, height, width, xy, 1]
                rot_normals = tf.matmul(rot_matrices, normals_xy)
                images = tf.concat((rot_normals[..., :2], images[..., 2:]), axis=-1)

            if tf.random.uniform(()) > 0.5:  # Randomly flop the image
                images = tf.image.flip_left_right(images)
                segmentations = tf.image.flip_left_right(segmentations)
                masks = tf.image.flip_left_right(masks)
                if self.rotate_normals:
                    images *= [[[-1, 1, 1]]]

        if self.small_rotate_inputs:
            logging.info("Adding small random image rotations of +-4deg")
            angles_rad = (tf.random.uniform((images.shape[0],)) * 0.13962634) - 0.06981317
            images = tfa.image.rotate(images, angles_rad, interpolation='BILINEAR', name="rotate_images")
            segmentations = tfa.image.rotate(segmentations, angles_rad, interpolation='NEAREST', name="rotate_segmentation")
            masks = tfa.image.rotate(masks, angles_rad, interpolation='NEAREST', name="rotate_masks")

        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            predictions = logits * segmentations

            regularization_loss = 0
            defect_loss = self.loss_object(masks, predictions)
            if len(self.model.losses) != 0:
                regularization_loss += tf.add_n(self.model.losses)

            loss = defect_loss + regularization_loss

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.lr.assign(lr)
        self.optimizer.apply_gradients(zip(gradients, variables))

    @tf.function
    def test_step(self, images, segmentations, masks):
        """
        Test detector network given the batch of images, foreground segmentations, and masks
        :param images: Batch of input images
        :param segmentations: Batch of foreground segmentation masks
        :param masks: Batch of defect segmentation masks
        :returns loss: Test loss for the batch
        :returns predictions: Predictions from the detector
        """
        segmentations = tf.cast(segmentations, tf.float32)
        masks = tf.cast(masks, tf.float32)

        predictions = self.model(images, training=False)
        predictions *= segmentations

        loss = self.loss_object(masks, predictions)

        return loss, predictions

    def train(self, train_data: DataIterator, test_data: DataIterator, nr_epochs: int, exp_name: str = None,
              logdir: str = 'logs', modeldir: str = 'models', test_epoch: int = 20, image_epoch: int = 100,
              initial_epoch_nr: int = 0, save_every: int = None, train_data_to_test: DataIterator = None):
        """
        Train the detector model on the train data, test it on test data and "train data to test".
        Log and save the results at selected intervals.
        :param train_data: Train data iterator
        :param test_data: Test data iterator
        :param nr_epochs: Number of training epochs (passes over entire train data)
        :param exp_name: String appended to the experiment name
        :param logdir: Path to the directory, where the tf.summary logfiles will be stored
        :param modeldir: Path to the directory, where the trained keras model will be stored
        :param test_epoch: Specifies interval of model testing on test data an "train data to test"
        :param image_epoch: Specifies interval of saving the images summary into the log file (has to be divisible by test_epoch)
        :param initial_epoch_nr: If not training from scratch, specify the number of the initial epoch (for log purporses)
        :param save_every: Specifies interval of saving the model to the modeldir
        :param train_data_to_test: Specifies another DataIterator on which the model is tested along the test data
        """
        tf.keras.backend.clear_session()

        if save_every is None:
            save_every = test_epoch

        basename = self.detector.get_name() + "-" + train_data.get_name()
        exp_name = Trainer.append_opt_string(basename, self.opt_kwargs, train_data.batch_size) + exp_name
        logging.info(exp_name)

        self.metrics['test'] = Metrics("test")
        if train_data_to_test is not None:
            self.metrics['train_to_test'] = Metrics("train_to_test")

        # Create summary file
        log_filename = path.join(logdir, exp_name)
        summary_writer = tf.summary.create_file_writer(log_filename)
        summary_writer.set_as_default()

        if "learning_rate" in self.opt_kwargs:
            learning_rate = self.opt_kwargs["learning_rate"]
        else:
            learning_rate = 0.001

        for epoch in tqdm(range(initial_epoch_nr, nr_epochs)):
            ti = time()
            is_test_epoch = (epoch + 1) % test_epoch == 0
            is_image_epoch = (epoch + 1) % image_epoch == 0
            is_save_epoch = (epoch + 1) % save_every == 0

            for i, (images, masks, segmentations, _, _) in enumerate(train_data):
                iteration = (epoch * train_data.length) + i
                lr = self.compute_ramped_lrate(iteration, nr_epochs * train_data.length, 0.1, 0.3, learning_rate)
                lr = tf.convert_to_tensor(lr, dtype=tf.float32)

                self.train_step(images, segmentations, masks, lr)

            if is_test_epoch:
                # Test on train set
                if train_data_to_test is not None:
                    if train_data_to_test.patch_shape != train_data_to_test.image_shape:
                        self._test_patched(train_data_to_test, epoch, is_image_epoch, prefix='train_to_test')
                    else:
                        self._test(train_data_to_test, epoch, is_image_epoch, prefix='train_to_test')

                # Test on test set
                if test_data.patch_shape != test_data.image_shape:
                    self._test_patched(test_data, epoch, is_image_epoch, prefix='test')
                else:
                    self._test(test_data, epoch, is_image_epoch, prefix='test')

                if (modeldir is not None) and is_save_epoch:
                    self.save(modeldir, exp_name, epoch)

        return exp_name

    def _test(self, test_data: DataIterator, epoch: int, log_images: bool, save_path: str = None, prefix: str = 'test'):
        """
        Tests the method on the test_data iterator, logs the metrics with prefix specified
        and saves the images if save_path is specified.
        Note: this method assumes the metric objects and log objects to be already created.
              If this is not the case use test() instead.
        :param test_data: test data iterator
        :param epoch: Epoch number used for logging
        :param log_images: Specifies whether to save images to the log file
        :param save_path: If specified, the predicted images are saved to this path
        :param prefix: Prefix used for logged metrics and images
        :returns: AuROC, AuPRC, Loss
        """

        for k, (images, masks, segmentations, _, locs) in enumerate(test_data):
            loss, predictions = self.test_step(images, segmentations, masks)

            t_predictions = tf.cast(predictions > 0.5, dtype=masks.dtype)
            self.metrics[prefix].update_t(loss, predictions, masks, t_predictions)

            if k == 0 and log_images:
                imsum = Trainer.create_defect_image_summary(images, predictions, masks)
                tf.summary.image(prefix + '/predictions', imsum, step=epoch, max_outputs=3)

            if save_path is not None:
                for prediction, loc in zip(predictions, locs):
                    bn = path.basename(test_data.filenames[loc[0]].seg_filename)
                    np.save(path.join(save_path, bn[:-17] + "_prediction.npy"), prediction)
                    prediction = Image.fromarray(np.squeeze((np.clip(prediction, 0, 1) * 255).astype(np.uint8), axis=2))
                    prediction.save(path.join(save_path, bn[:-17] + "_prediction.png"))

        auroc = self.metrics[prefix].auroc.result()
        auprc = self.metrics[prefix].auprc.result()
        dlloss = self.metrics[prefix].loss.result()

        self.metrics[prefix].write(epoch)
        self.metrics[prefix].reset()

        return auroc, auprc, dlloss

    def _test_patched(self, test_data: DataIterator, epoch: int, log_images: bool, save_path: str = None, prefix: str = 'test'):
        """
        Tests the method on the test_data iterator iterating over patches. Patches are combined into images based on
        coordinates returned in loc array from next().
        Logs the metrics with prefix specified and saves the images if save_path is specified.
        Note: this method assumes the metric objects and log objects to be already created.
              If this is not the case use test() instead.
        :param test_data: test data iterator
        :param epoch: Epoch number used for logging
        :param log_images: Specifies whether to save images to the log file
        :param save_path: If specified, the predicted images are saved to this path
        :param prefix: Prefix used for logged metrics and images
        :returns: AuROC, AuPRC, Loss
        """
        out_predictions = np.zeros((test_data.total_samples,) + test_data.image_shape + (1,), dtype=np.float32)
        out_predictions_t = np.zeros((test_data.total_samples,) + test_data.image_shape + (1,), dtype=np.bool)
        out_masks = np.zeros((test_data.total_samples,) + test_data.image_shape + (1,), dtype=np.bool)
        out_segmentations = np.zeros((test_data.total_samples,) + test_data.image_shape + (1,), dtype=np.bool)
        out_images = np.zeros((test_data.total_samples,) + test_data.image_shape + (test_data.total_illums,), dtype=np.float32)

        for k, (images, masks, segmentations, _, loc) in enumerate(test_data):
            self.combine_patches(masks, out_masks, test_data.patch_shape, test_data.test_overlap, loc)
            self.combine_patches(images, out_images, test_data.patch_shape, test_data.test_overlap, loc)
            self.combine_patches(segmentations, out_segmentations, test_data.patch_shape, test_data.test_overlap, loc)

            b_loss, predictions = self.test_step(images, segmentations, masks)

            t_predictions = tf.cast(predictions > 0.5, dtype=np.bool)

            self.combine_patches(predictions, out_predictions, test_data.patch_shape, test_data.test_overlap, loc)
            self.combine_patches(t_predictions, out_predictions_t, test_data.patch_shape, test_data.test_overlap, loc)

            self.metrics[prefix].update_t(b_loss, predictions, masks, t_predictions)

        if log_images:
            imsum = Trainer.create_defect_image_summary(out_images, out_predictions, out_masks)
            tf.summary.image(prefix + '/predictions', imsum, step=epoch, max_outputs=3)

        if save_path is not None:
            for i, prediction in enumerate(out_predictions):
                bn = path.basename(test_data.filenames[i].seg_filename)
                np.save(path.join(save_path, bn[:-17] + "_prediction.npy"), prediction)
                prediction = Image.fromarray(np.squeeze((np.clip(prediction, 0, 1) * 255).astype(np.uint8), axis=2))
                prediction.save(path.join(save_path, bn[:-17] + "_prediction.png"))

        auroc = self.metrics[prefix].auroc.result()
        auprc = self.metrics[prefix].auprc.result()
        dlloss = self.metrics[prefix].loss.result()

        self.metrics[prefix].write(epoch)
        self.metrics[prefix].reset()

        return auroc, auprc, dlloss

    def test(self, test_data, epoch, save_path=None, log_images=False, log_path=None, exp_name=None):
        """
        Tests the method on the test_data iterator, logs the metrics with prefix specified
        and saves the images if save_path is specified.
        :param test_data: test data iterator
        :param epoch: Epoch number used for logging
        :param save_path: If specified, the predicted images are saved to this path in folder specified by exp_name
        :param log_images: Specifies whether to save images to the log file
        :param log_path: If specified, the metrics are logged into this directory in folder specified by exp_name
        :param exp_name: Experiment name used for logging and saving. If None, datetime is used
        :returns: AuROC, AuPRC, meanIoU, bestIoU, best_meanIoU, Loss
        """

        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d%H%M%S")

        self.metrics['test'] = Metrics("test")
        if log_path is not None:
            save_path = path.join(log_path, exp_name)
            summary_writer = tf.summary.create_file_writer(log_path)
            summary_writer.set_as_default()

        if save_path is not None:
            save_path = path.join(save_path, exp_name)
            if not path.exists(save_path):
                makedirs(save_path)

        auroc, auprc, dloss = self._test(test_data, epoch, log_images, save_path=save_path)

        if tf.is_tensor(auroc):
            auroc = auroc.numpy()
        if tf.is_tensor(auprc):
            auprc = auprc.numpy()
        if tf.is_tensor(dloss):
            dloss = dloss.numpy()

        return auroc, auprc, dloss

    def save(self, modeldir, exp_name, epoch):
        """
        Saves the trained model into the modeldir with the name equal to [exp_name]_e[epoch]_Det.h5
        :param modeldir: Path to the directory, where the trained keras model will be stored
        :param exp_name: Name of the experiment = basename of the stored model file
        :param epoch: None or integer epoch number that will be appended to the model filename
        """
        filename = exp_name + "_Det.h5" if epoch is None else exp_name + "_e{:d}_Det.h5".format(epoch)
        self.model.save(path.join(modeldir, filename), save_format="h5")

    def summary(self):
        """Prints the keras summary of the detector model"""
        self.model.summary()

    @staticmethod
    def create_defect_image_summary(images, predictions, masks, max_nr=3):
        """
        Combines the images, predictions, and masks to form a image summary
        :param images: Batch of input images
        :param predictions: Batch of predicted defect segmentation masks
        :param masks: Batch of ground truth defect segmentation masks
        :param max_nr: Maximal number of summary images from this batch
        :returns Summary image
        """
        images = images[:max_nr]
        predictions = predictions[:max_nr]
        masks = masks[:max_nr]

        if images.shape[-1] > 12:
            images = np.std(images, axis=-1, keepdims=True)
        else:
            images = images[..., 0:1]
        images -= np.amin(images, axis=(1, 2), keepdims=True)
        images /= np.amax(images, axis=(1, 2), keepdims=True)
        masks = np.cast[np.float32](masks)

        out_comb = np.concatenate((images, masks, predictions), axis=2)

        return np.clip(out_comb, 0, 1)

    @staticmethod
    def append_opt_string(base_name, opt_kwargs, batch_size):
        """
        Appends a string specifying the optimization settings to the base_name
        :param base_name: Experiment base name
        :param opt_kwargs: dictionary containing optimizer args {optimizer: 'adam' or 'SGD', learning_rate:0.001, ... }
        :param batch_size: Batch size
        :returns: Experiment name extended by the optimization settings
        """

        if "optimizer" in opt_kwargs and (opt_kwargs["optimizer"] is not None or opt_kwargs["optimizer"] != "adam"):
            opt = opt_kwargs["optimizer"]
        else:
            opt = ""

        if "learning_rate" in opt_kwargs and opt_kwargs["learning_rate"] != 0.001:
            lr = "-lr" + str(opt_kwargs["learning_rate"])
        else:
            lr = ""

        return base_name + "{opt}{lr}-b{batch}".format(
            opt=opt, lr=lr, batch=batch_size,
        )

    @staticmethod
    def compute_ramped_lrate(i, iteration_count, ramp_up_fraction, ramp_down_fraction, learning_rate):
        """
        Source: https://github.com/NVlabs/selfsupervised-denoising/blob/master/selfsupervised_denoising.py
        Ramp up the learning rate in the first ramp_up_fraction of iteration_count using cosine schedule
        Ramp down the learning rate in the last ramp_down_fraction of iteration_count using cosine schedule
        :param i: Current iteration
        :param iteration_count: Total number of iterations
        :param ramp_up_fraction: Ratio of iterations where to apply the cosine schedule to increase the learning rate
        :param ramp_down_fraction: Ratio of iterations where to apply the cosine schedule to decrease the learning rate
        :param learning_rate: The base learning rate used in the schedule
        :returns: learning rate for the iteration i
        """
        if ramp_up_fraction > 0.0:
            ramp_up_end_iter = iteration_count * ramp_up_fraction
            if i <= ramp_up_end_iter:
                t = (i / ramp_up_fraction) / iteration_count
                learning_rate = learning_rate * (0.5 - np.cos(t * np.pi) / 2)

        if ramp_down_fraction > 0.0:
            ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
            if i >= ramp_down_start_iter:
                t = ((i - ramp_down_start_iter) / ramp_down_fraction) / iteration_count
                learning_rate = learning_rate * (0.5 + np.cos(t * np.pi) / 2) ** 2

        return learning_rate

    @staticmethod
    def combine_patches(patches, target, patch_shape, overlap, locations, channel=None):
        """
        Add the patches to the target array based on the coordinates stored in locations
        :param patches: Image patches with shape patch_shape
        :param target: Target image array of shape image_shape
        :param patch_shape: Shape of the patches
        :param overlap: Ratio of patch overlap [0-1]
        :param locations: Patch coordinates in the form [[sample_id, y_coord, x_coord],...]
        :param channel: If specified, only the channel with this index is used
        :returns: Target array with patches filled in
        """
        overlap1 = int((patch_shape[0] * overlap) // 2)
        overlap2 = int((patch_shape[1] * overlap) // 2)

        for j, loc in enumerate(locations):
            if loc[1] == 0 and loc[2] == 0:
                if channel is None:
                    is1 = np.s_[loc[0], :patch_shape[0], :patch_shape[1], ...]
                else:
                    is1 = np.s_[loc[0], :patch_shape[0], :patch_shape[1], ..., channel]
                isp = np.s_[j, :, :, ...]
            elif loc[1] == 0:
                if channel is None:
                    is1 = np.s_[loc[0], :patch_shape[0], loc[2] + overlap2:loc[2] + patch_shape[1], ...]
                else:
                    is1 = np.s_[loc[0], :patch_shape[0], loc[2] + overlap2:loc[2] + patch_shape[1], ..., channel]
                isp = np.s_[j, :, overlap1:, ...]
            elif loc[2] == 0:
                if channel is None:
                    is1 = np.s_[loc[0], loc[1] + overlap1:loc[1] + patch_shape[0], :patch_shape[1], ...]
                else:
                    is1 = np.s_[loc[0], loc[1] + overlap1:loc[1] + patch_shape[0], :patch_shape[1], ..., channel]
                isp = np.s_[j, overlap1:, :, ...]
            else:
                if channel is None:
                    is1 = np.s_[loc[0], loc[1] + overlap1:loc[1] + patch_shape[0],
                                loc[2] + overlap2:loc[2] + patch_shape[1], ...]
                else:
                    is1 = np.s_[loc[0], loc[1] + overlap1:loc[1] + patch_shape[0],
                                loc[2] + overlap2:loc[2] + patch_shape[1], ..., channel]
                isp = np.s_[j, overlap1:, overlap1:, ...]

            target[is1] = patches[isp]


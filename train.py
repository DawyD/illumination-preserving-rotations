import numpy as np
import tensorflow as tf

import os
import logging

from tensorflow.keras.callbacks import ModelCheckpoint
from callbacks.testandlog import TestAndLogCallback
from callbacks.cosinerampscheduler import CosineRampLearningRateScheduler
from data.dataloader import DataIterator

import argparse
import custom_trainer
from models.unet import UNet
from models.dncnn import DnCNN
from models.iternet import IterNet
from models.fcn import FCN
from models.deeplabv3plus import DeepLabV3plus

parser = argparse.ArgumentParser(description='Photometric stereo segmentation')
parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
parser.add_argument('--epochs', '-e', type=int, default=5000, help='Number of epochs')
parser.add_argument('--nr_illums', '-i', type=int, default=24, help='Number of illuminations')
parser.add_argument('--rotations', '-K', type=str, default="rot30",
                    help="Augment data by rotation: None - no rotations, 'rotRND' - random rotation,"
                         "'rot30','rot60','rot90', or 'rot180' - rotation by integer multiples of the angle selected")
parser.add_argument('--no_illum_pres', action='store_true', help="Do not use illumination preserving rotations")
parser.add_argument('--arch', '-a', type=str, default="unet",
                    help='Architecture type [unet, dncnn, iternet, fcn, deeplabv3]')
parser.add_argument('--dataset_path', '-p', type=str, default=None, help='Dataset path')
parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size')
parser.add_argument('--suffix', type=str, default="", help='Suffix to the name under which the models will be saved')
parser.add_argument('--memory_limit', '-g', type=int, default=-1,
                    help='If > 0, GPU memory to allocate in MB on the first GPU.'
                         'If <= 0 all GPUs are fully allocated')
parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer")
parser.add_argument('--save_every', '-v', type=int, default=5000, help='Save model every [V] epochs')
parser.add_argument('--use_patches', action='store_true', help="Use rather patch_size=128x128, and batch norm")
parser.add_argument('--dim_reduction', type=str, default=None,
                    help="None - no reduction, subset of 'albedo,normals' for Lambertian reduction, "
                         "subset of 'mean,std,skew' for statistical moments reduction")
parser.add_argument('--drop_rate', type=float, default=0, help='Dropout rate to be used in every layer')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L1 weight decay rate')
parser.add_argument('--keras_only', action='store_true', help="Use keras.fit() instead of the custom training loop")
args = parser.parse_args()

if ((args.drop_rate != 0) or (args.weight_decay != 1e-5)) and (args.arch != "unet"):
    logging.warning("The weight_decay and drop_rate arguments are ignored for architectures other than unet")

''' Fix the random seeds '''

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

''' Setup the GPU '''

logging.info("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus and args.memory_limit > 0:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)])

''' Load data '''

patch_shape = (128, 128) if args.use_patches else (512, 512)
batch_size = args.batch_size
log_dir = "logs"
save_dir = "checkpoints"
epoch_multiplier = 100 if args.keras_only else 1

if args.dataset_path is None:
    raise ValueError("Dataset path has to be set")
if not os.path.exists(args.dataset_path):
    raise ValueError("Dataset path does not exist")

train_data = DataIterator(base_path=os.path.join(args.dataset_path, "Train"),
                          sample_nrs=None, illum_nrs=args.nr_illums,
                          batch_size=args.batch_size, patch_shape=patch_shape, image_shape=(512, 512), channels=1,
                          dim_reduction=args.dim_reduction, quiet=False, keras_iterator=args.keras_only,
                          aug_rotation=args.rotations if args.keras_only or (args.rotations != "rotRND") else None,
                          preserve_illumination_channels=(not args.no_illum_pres),
                          train_mode=True, aug_shift_bright=True, epoch_multiplier=epoch_multiplier)

test_data = DataIterator(base_path=os.path.join(args.dataset_path, "Test"),
                         sample_nrs=None, illum_nrs=args.nr_illums,
                         batch_size=args.batch_size, patch_shape=(512, 512), image_shape=(512, 512), channels=1,
                         dim_reduction=args.dim_reduction, quiet=False, keras_iterator=args.keras_only,
                         aug_rotation=None, preserve_illumination_channels=False,
                         train_mode=False, aug_shift_bright=False)

train_data_to_test = DataIterator(base_path=os.path.join(args.dataset_path, "Train"),
                                  sample_nrs=None, illum_nrs=args.nr_illums, batch_size=args.batch_size,
                                  patch_shape=(512, 512), image_shape=(512, 512), channels=1,
                                  dim_reduction=args.dim_reduction, quiet=False, keras_iterator=args.keras_only,
                                  aug_rotation=None, preserve_illumination_channels=False,
                                  train_mode=False, aug_shift_bright=False)

valid_data = None  # Can be used e.g. for early stopping

''' Define the architecture '''

if args.use_patches:
    input_shape = (None, None, train_data.sample_channels,)
else:
    input_shape = patch_shape + (train_data.sample_channels,)

if args.arch == "unet":
    arch = UNet(input_shape,
                out_channels=1,
                nr_feats=64,
                nr_blocks=4,
                nr_conv=2,
                upscale="nn",
                drop_rate=args.drop_rate,
                last_activation="sigmoid",
                normalization="group" if not args.use_patches else "batch",
                nr_groups=-1,
                conv_type="full",
                name="Unet",
                initializer="truncated_normal",
                weight_decay=args.weight_decay,
                weight_decay_type="l1")
elif args.arch == "small":
    arch = DnCNN(input_shape,
                 out_channels=1,
                 nr_feats=18,
                 nr_conv=2,
                 last_activation="sigmoid",
                 normalization="group" if not args.use_patches else "batch",
                 nr_groups=-1,
                 conv_type="full",
                 name="DnCNN",
                 weight_decay=None,
                 weight_decay_type=None)
elif args.arch == "iternet":
    arch = IterNet(input_shape,
                   nr_feats=32,
                   drop_rate=0.1,
                   iteration=3)
elif args.arch == "fcn":
    arch = FCN(input_shape,
               weight_decay=5e-4,
               drop_rate=0.5)
elif args.arch == "deeplabv3+":
    arch = DeepLabV3plus(
        input_shape,
        weight_decay=None, weight_decay_type=None,
        training=None,
        backbone="xception",
        alpha=1,
        OS=8)
else:
    raise ValueError("Unknow architecture")

if args.keras_only:
    ''' Define the metrics, loss, and compile the model'''

    metrics = [
        tf.keras.metrics.AUC(curve='ROC', name='AuROC'),
        tf.keras.metrics.AUC(curve='PR', name='AuPRC'),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.BinaryCrossentropy()
    ]

    model = arch.net()
    model.compile(optimizer=args.optimizer, loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=metrics)
    model.summary()

    ''' Define the calllbacks '''

    model_name = "{arch:}{suffix:}{data:}_s{seed:}_B{batch:}".format(
        arch=arch.get_name(), suffix=args.suffix, data=train_data.get_name(), seed=args.seed, batch=args.batch_size)

    model_chackpoint = ModelCheckpoint(
        filepath=save_dir + "/" + model_name,
        save_freq=len(train_data) * args.save_every // epoch_multiplier)

    test_and_log = TestAndLogCallback(
        [train_data_to_test, test_data, valid_data],
        ["train_to_test", "test", "valid"],
        verbose=False, log_dir=log_dir + "/" + model_name,
        scalar_freq=100,
        image_freq=200,
        channels=3 if args.dim_reduction is not None else None,
        epoch_multiplier=epoch_multiplier)

    lr_scheduler = CosineRampLearningRateScheduler(
        total_number_of_batches=len(train_data) * (args.epochs // epoch_multiplier),
        ramp_up_fraction=0.1,
        ramp_down_fraction=0.3,
        learning_rate=0.001)

    ''' Train '''

    model.fit(train_data, epochs=args.epochs // epoch_multiplier, verbose=1, initial_epoch=0, validation_data=test_data,
              validation_freq=(100 // epoch_multiplier), callbacks=[model_chackpoint, test_and_log, lr_scheduler])

    ''' Save the final model '''

    model.save("checkpoints/" + model_name)

else:
    trainer_obj = custom_trainer.Trainer(
        arch,
        rotate_inputs=False if args.rotations != "rotRND" else "rotRND",
        rotate_normals=True if (args.dim_reduction is not None) and ("normals" in args.dim_reduction) else False,
        small_rotate_inputs=False,
        opt_kwargs={'optimizer': args.optimizer})

    trainer_obj.summary()

    exp_name = trainer_obj.train(train_data, test_data, train_data_to_test=train_data_to_test,
                                 nr_epochs=args.epochs, test_epoch=100, image_epoch=100, save_every=1000,
                                 logdir=log_dir, modeldir=save_dir, exp_name=args.suffix)


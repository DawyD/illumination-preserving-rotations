import os

import numpy as np
from PIL import Image
import time
from os import path
from collections import namedtuple
import cv2  # for rotations
from scipy.stats import skew  # for dimensionality reduction
from scipy.io import loadmat
from glob import glob
import math

from tensorflow.keras.utils import Sequence

import logging


class DataIteratorNormals(Sequence):
    def __init__(self, base_path, sample_nrs=None,
                 batch_size=2, patch_shape=(512, 512), image_shape=(512, 512),
                 train_mode=False, quiet=False, aug_rotation=None, aug_shift=False,
                 keras_iterator=True, epoch_multiplier=1):
        """
        Data generator for multi-illumination image stacks.
        It loads all the samples specified in sample_nrs from the base_path to the memory
        It specifies augmentation options applied in each iteration.

        :param base_path: Path to the dataset
        :param sample_nrs: List of sample numbers to be loaded (e.g. ['defect_001', '026', 'defect_002'])
        :param batch_size: Number of samples in a batch returned by the generator (e.g. 2)
        :param patch_shape: Shape of patch returned by the generator. If smaller than image_shape a crop is returned.
               (e.g. (512, 512))
        :param image_shape: Shape of source images in the dataset (e.g. (512, 512))
        :param train_mode: If true, randomly crops and augments the images, If false, all crops are selected
        :param quiet:
        :param aug_rotation: Augment data by rotation: None - no rotations, 'rotRND' - random rotation,
                             'rot30','rot60','rot90', or 'rot180' - rotation by integer multiples of the angle selected
        :param aug_shift: Augment data by shifting images by up to +-4px
                                 for uint8 images or by +-0.02 if some dimensionality reduction is applied
        :param keras_iterator: If True, the iterator returns patches, masks, segmentations
                               If False, it also returns whether it has annotations, and locations
        :param epoch_multiplier: In training mode multiplies the length of an epoch
        """

        # Settings for loading the images
        self.channels = 3
        self.image_shape = image_shape
        self.sample_nrs = sample_nrs if sample_nrs is not None else [os.path.basename(p)[:-4] for p in glob(base_path + "/*.mat")]
        print("Samples:", self.sample_nrs)
        self.total_samples = len(self.sample_nrs)

        self.patch_shape = patch_shape
        self.masks_ch_ids = [self.channels, self.channels + 1]

        # Generator setting
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.test_overlap = 0.5
        self.keras_iterator = keras_iterator
        if self.keras_iterator:
            self.iter_tuple = namedtuple("data", ["patches", "masks", "segmentations"])
        else:
            self.iter_tuple = namedtuple("data", ["patches", "masks", "segmentations", "has_annotation", "locations"])
        self.epoch_multiplier = epoch_multiplier

        # Augmentation settings:
        self.aug_rotation = aug_rotation
        self.aug_shift = aug_shift

        self.augment = (not self.aug_shift) \
                       and (self.aug_rotation is None) \
                       and (self.patch_shape == self.image_shape)

        if (aug_rotation is not None) and (aug_rotation.lower() == "none"):
            aug_rotation = None

        if not ((aug_rotation == 'rot30') or (aug_rotation == 'rot60') or (aug_rotation == 'rot90')
                or (aug_rotation == 'rot180') or (aug_rotation == 'rotRND') or (aug_rotation is None)):
            raise ValueError("aug_rotation has to be 'rot30','rot60','rot90', 'rot180', 'rotRND', or None")

        if self.aug_rotation == "rot30":
            self.prerotate_angle = 30
        elif self.aug_rotation == "rot60":
            self.prerotate_angle = 60
        else:
            self.prerotate_angle = 0

        if (self.aug_rotation == "rot30") or (self.aug_rotation == "rot90"):
            self.illum_pres_rot_angle = 90
        elif (self.aug_rotation == "rot60") or (self.aug_rotation == "rot180"):
            self.illum_pres_rot_angle = 180
        else:
            self.illum_pres_rot_angle = 0

        start = time.time()

        # Load images [batch_size, crop_height, crop_width, channels, input_channels]
        self.data, self.has_annotations = self._load_data(base_path)

        self.length = self.total_samples * (self.data.shape[1] // patch_shape[0]) * \
                                           (self.data.shape[2] // patch_shape[1])

        # Pre-generate the list of test locations [sample_id, crop description, ....] for one epoch
        # (training locations are generated randomly)
        if train_mode:
            self.locations = self._get_train_locations()
        else:
            self.locations = self._load_test_locations()

        if not quiet:
            logging.info("Time of loading: {:0.2f}s".format(time.time() - start))
            details_str = "Raw {ar:} with shape {sh:} of size {sze:} MB of type {dt} in range [{min:0.1f}, {max:0.1f}]"
            logging.info(details_str.format(
                ar="images", sh=self.data[..., :-2].shape, sze=self.data[..., :-2].nbytes / 1024 / 1024,
                min=np.amin(self.data[..., :-2]), max=np.amax(self.data[..., :-2]), dt=self.data[..., :-2].dtype))
            logging.info(details_str.format(
                ar="segmentations", sh=self.data[..., -1:].shape, sze=self.data[..., -1:].nbytes / 1024 / 1024,
                min=np.amin(self.data[..., :-1:]), max=np.amax(self.data[..., :-1]), dt=self.data[..., :-1].dtype))
            logging.info(details_str.format(
                ar="masks", sh=self.data[..., -2:-1].shape, sze=self.data[..., -2:-1].nbytes / 1024 / 1024,
                min=np.amin(self.data[..., -2:-1]), max=np.amax(self.data[..., -2:-1]), dt=self.data[..., -2:-1].dtype))
            logging.info("Epoch size: %d" % self.length)
            if not train_mode:
                logging.info("Test_locations: %d" % len(self.locations))

            if self.augment:
                logging.info("Augmentations: None")
            else:
                logging.info("Augmentations:")
                logging.info("  Rotation:", self.aug_rotation)
                logging.info("  Shifts:", self.aug_shift)

    def __getitem__(self, idx):
        """
        Returns a single batch of images, masks and other properties
        :return images in format [batch_size, height, width, channels, nr_views]
        :return segmentations in format [batch_size, height, width, 1]
        :return masks in format [batch_size, height, width, 1] or None if get_masks = False
        :return has_annotation in format [batch_size, 1]
        :return locations in format [batch_size, sample_nr + x_coordinate + y_coordinate]
        """
        locations = self.locations[idx * self.batch_size:(idx + 1) * self.batch_size]

        patches = self._get_patch(locations)
        imgs, segmentations, masks = self._split(patches)

        if self.keras_iterator:
            return self.iter_tuple(imgs, masks, segmentations)
        else:
            has_annotation = self._get_props(self.has_annotations, locations)
            return self.iter_tuple(imgs, masks, segmentations, has_annotation, locations)

    def __len__(self):
        return math.ceil(len(self.locations) / self.batch_size)

    def on_epoch_end(self):
        if self.train_mode:
            self.locations = self._get_train_locations()

    def get_sample_nr(self, sample_id):
        """
        Returns a name of a sample with the given sample id
        :param sample_id: Sample ID
        """

        sample_id %= self.total_samples
        return self.sample_nrs[sample_id]

    def _split(self, patches):
        """
        Splits the data cube into images, foreground segmentation mask, and defect segmentation mask.
        Transforms the images into defined range by self.norm, and transforms masks to boolean data type
        :param patches: The data cube
        :return: Multi-illumination image stack (float if self.norm is not None)
        :return: Boolean foreground segmentation mask
        :return: Boolean defect segmentation mask
        """
        images = patches[..., :self.channels]
        segmentations = patches[..., -1][..., None] > 0
        masks = patches[..., -2][..., None] > 0

        return images * segmentations, segmentations, masks
    
    @staticmethod
    def _rotate_vectors(vectors, angle):
        """
        Rotates the vector of shape [n,3] by an angle along X,Y coordinates counter-clockwise
        :param vectors: Array of shape [n, 3] where each one of n vectors is (x,y,z)
        :param angle: Angle in degrees by which rotate the vectors
        :return: Rotated copy of the vectors
        """
        if angle == 0:
            return vectors
        rot_vectors = vectors.copy()
        rad_rot = -np.deg2rad(angle)
        rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                 (np.sin(rad_rot),  np.cos(rad_rot))], axis=0)  # [x, y]
        rot_vectors[:, :2] = np.matmul(rot_vectors[:, :2], rot_matrices)
        return rot_vectors

    def _load_data(self, base_path):
        """
        Loads the images and masks into the memory
        :param base_path: Path to the dataset
        :return: numpy array containing all the images and masks after the possible dimensionality reduction
        """

        # Check the directory

        # Allocate the data
        mask_channels = 2
        # In case of rotation by 30 or 60, preload the rotated versions to memory
        total_nr_samples = self.total_samples if self.prerotate_angle == 0 else self.total_samples*3

        data = np.zeros((total_nr_samples,) + self.image_shape + (self.channels + mask_channels,), dtype=np.float32)
        has_annotations = np.zeros((self.total_samples,), dtype=np.bool_)

        raw_sample_channels = self.channels + mask_channels
        # For each sample
        for j, sample_nr in enumerate(self.sample_nrs):
            seg_filename = path.join(base_path, sample_nr + "_segmentation.png")
            mask_filename = path.join(base_path, sample_nr + "_mask.png")
            
            # Load foreground segmentation mask
            with Image.open(seg_filename) as seg:
                sampledata = np.zeros(seg.size[::-1] + (raw_sample_channels,), dtype=np.float32)
                if seg.mode == "LA":
                    sampledata[..., -1] = (np.array(seg) > 0).astype(np.float32)[..., 0]
                else:
                    sampledata[..., -1] = (np.array(seg) > 0).astype(np.float32)

            # Load defect segmentation mask
            if mask_filename is not None and path.isfile(mask_filename):
                with Image.open(mask_filename) as maskf:
                    if maskf.mode == "LA":
                        sampledata[..., -2] = (np.array(maskf) > 0).astype(np.float32)[..., 0]
                    else:
                        sampledata[..., -2] = (np.array(maskf) > 0).astype(dtype=np.float32)

                has_annotations[j] = True

            # Load stack of images

            sampledata[..., :self.channels] = loadmat(path.join(base_path, sample_nr + ".mat"))["Normal_est"]

            for k in range(1 if self.prerotate_angle == 0 else 3):
                rot_sampledata = self.rotate(sampledata, k*self.prerotate_angle, interp="bicubic")

                # Rotate the XY normals
                rad_rot = -np.deg2rad(k*self.prerotate_angle)
                rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                         (np.sin(rad_rot), np.cos(rad_rot))], axis=0)  # [x, y]
                rot_sampledata[..., :2] = np.matmul(rot_sampledata[..., :2], rot_matrices)

                data[j + k*self.total_samples] = rot_sampledata

        return data, has_annotations
    def augment_fn(self, in_array, do_shift,
                   illum_pres_rot_angle, do_arb_rotation, bb):
        in_array = np.copy(in_array)

        if illum_pres_rot_angle != 0:
            if illum_pres_rot_angle == 90:
                k_rot = np.random.randint(4)
            elif illum_pres_rot_angle == 180:
                k_rot = np.random.randint(2) * 2
            else:
                raise ValueError("Only 90 and 180 degrees are supported")

            in_array = np.rot90(in_array, k=k_rot)

            rad_rot = -np.deg2rad(90 * k_rot)
            rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                     (np.sin(rad_rot), np.cos(rad_rot))], axis=0)  # [x, y]

            in_array[..., :2] = np.matmul(in_array[..., :2], rot_matrices)

        if do_arb_rotation:
            angle = np.random.randint(0, 360)
            if angle < 0:
                in_array = np.fliplr(in_array)
            in_array = self.rotate(in_array, np.abs(angle))

            rad_rot = -np.deg2rad(np.abs(angle))
            rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                     (np.sin(rad_rot), np.cos(rad_rot))], axis=0)  # [x, y]

            in_array[..., :2] = np.matmul(in_array[..., :2], rot_matrices)

            if angle < 0:
                in_array *= [[[-1, 1, 1]]] # = np.concatenate((-in_array[..., :1], in_array[..., 1:]), axis=-1)

        if do_shift:
            shift = np.random.randint(-4, 5, size=2)

            new_image = np.zeros_like(in_array)
            if shift[0] == 0:
                if shift[1] == 0:
                    new_image = in_array
                elif shift[1] > 0:
                    new_image[:, shift[1]:, :] = in_array[:, :-shift[1], :]
                else:
                    new_image[:, :shift[1], :] = in_array[:, -shift[1]:, :]
            elif shift[0] > 0:
                if shift[1] == 0:
                    new_image[shift[0]:, :, :] = in_array[:-shift[0], :, :]
                elif shift[1] > 0:
                    new_image[shift[0]:, shift[1]:, :] = in_array[:-shift[0], :-shift[1], :]
                else:
                    new_image[shift[0]:, :shift[1], :] = in_array[:-shift[0], -shift[1]:, :]
            else:  # shift[0] < 0
                if shift[1] == 0:
                    new_image[:shift[0], :, :] = in_array[-shift[0]:, :, :]
                elif shift[1] > 0:
                    new_image[:shift[0], shift[1]:, :] = in_array[-shift[0]:, :-shift[1], :]
                else:
                    new_image[:shift[0], :shift[1], :] = in_array[-shift[0]:, -shift[1]:, :]

            in_array = new_image

        return in_array[bb[0]:bb[0] + self.patch_shape[0], bb[1]:bb[1] + self.patch_shape[1], :]

    def _get_props(self, per_sample_array, locations):
        """
        Based on sample id from locations it returns particular elements from per_sample_array
        :param per_sample_array: Array of the same length as self.sample_nrs
        :param locations: Array of locations Location = [sample_nr, x_coordinate, y_coordinate]
        :return: None if per_sample_array is None, else returns selected elements from per_sample_array
        """
        if per_sample_array is not None:
            return np.array([per_sample_array[loc[0] % self.total_samples] for loc in locations])
        else:
            return None

    def _get_patch(self, locations):
        """
        Extracts patches from self.data based on the locations
        :param locations: Array of locations. (Location=[sample_nr, x_coordinate, y_coordinate])
        :return: array of patches of size [batch_size, patch_shape[0], patch_shape[1], self.data.shape[-1]]
        """
        if self.augment:
            out_array = self.data[locations[:, 0]]
        else:
            out_array = np.empty((len(locations), self.patch_shape[0], self.patch_shape[1],
                                  self.data.shape[-1]), dtype=self.data.dtype)
            for j, loc in enumerate(locations):
                out_array[j] = self.augment_fn(self.data[loc[0]],
                                               do_shift=self.aug_shift,
                                               illum_pres_rot_angle=self.illum_pres_rot_angle,
                                               do_arb_rotation=self.aug_rotation == "rotRND",
                                               bb=(loc[1], loc[2]))

        return out_array

    def _get_train_locations(self):
        """
        Gets locations of the patches for the training batch. Location = [sample_nr, x_coordinate, y_coordinate]
        :return: array of locations
        """

        training_length = self.total_samples * (self.data.shape[1] // self.patch_shape[0]) * \
                                               (self.data.shape[2] // self.patch_shape[1]) * self.epoch_multiplier

        sample_ids = np.random.randint(0, len(self.data), training_length)

        if self.patch_shape[:2] != self.data.shape[1:3]:
            max_x = self.data.shape[2] - self.patch_shape[1] + 1
            max_y = self.data.shape[1] - self.patch_shape[0] + 1
            toplefts = np.random.randint(0, max_y * max_x, training_length)
            y, x = np.unravel_index(toplefts, (max_y, max_x))
        else:
            x = np.zeros_like(sample_ids)
            y = x

        loc = np.stack((sample_ids, y, x), axis=1)

        return loc

    def _load_test_locations(self):
        """
        Returns the locations for all samples so that it spatially covers the entire input images in case of cropping
        Location = [sample_nr, x_coordinate, y_coordinate]
        """
        test_locations = []
        for y in range(0, self.data.shape[1] - self.patch_shape[0] - 1, int(self.patch_shape[0] * self.test_overlap)):
            for x in range(0, self.data.shape[2] - self.patch_shape[1] - 1,
                           int(self.patch_shape[1] * self.test_overlap)):
                test_locations.append((x, y))

            # Append right border
            test_locations.append((self.data.shape[2] - self.patch_shape[1], y))

        # Append bottom border
        for x in range(0, self.data.shape[2] - self.patch_shape[1], int(self.patch_shape[1] * self.test_overlap)):
            test_locations.append((x, self.data.shape[1] - self.patch_shape[0]))

        # Append bottom-right patch
        test_locations.append((self.data.shape[2] - self.patch_shape[1], self.data.shape[1] - self.patch_shape[0]))

        test_locations = np.array(test_locations)

        return np.array([np.array(([i, t[0], t[1]])) for i in range(len(self.data)) for t in test_locations])

    @staticmethod
    def rotate(in_array, angle, interp="bicubic"):
        """
        Rotates the in_array "image" around its XY centre counter-clockwise
        :param in_array: Array of shape [height, width, channels]
        :param angle: Angle of rotation in the counter-clockwise direction
        :param interp: Interpolation, either 'nearest', 'bilinear', or 'bicubic'
        :return: Rotated array
        """
        if angle == 0:
            return in_array
        
        if interp == "nearest":
            flag = cv2.INTER_NEAREST
        elif interp == "bicubic":
            flag = cv2.INTER_CUBIC
        elif interp == "bilinear":
            flag = cv2.INTER_LINEAR
        else:
            raise ValueError("Unsupported interpolation")

        rows, cols = in_array.shape[0], in_array.shape[1]
        rot_matrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)

        former_type = in_array.dtype
        former_shape = in_array.shape

        if in_array.dtype == np.bool_:
            in_array = np.cast[np.uint8](in_array)

        if len(in_array.shape) > 3:
            in_array = in_array.reshape(in_array.shape[:2] + (-1,))
        elif len(in_array.shape) == 2:
            in_array = in_array[..., np.newaxis]

        out = np.empty(in_array.shape, dtype=in_array.dtype)
        for i in range(in_array.shape[-1]):
            out[..., i] = cv2.warpAffine(in_array[..., i], rot_matrix, (cols, rows), flags=flag)

        out = out.reshape(former_shape)

        if former_type == np.bool_:
            return out > 0.5
        else:
            return out

    def get_name(self):
        patch_shape = ""
        if self.patch_shape != (512, 512):
            patch_shape = "P" + str(self.patch_shape[0])

        return "{patch}K{rot}".format(
            patch=patch_shape,
            rot=self.aug_rotation,
        )

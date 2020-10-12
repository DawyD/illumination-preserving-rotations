import os

import numpy as np
from PIL import Image
import time
from os import path
from collections import namedtuple
import cv2  # for rotations
from scipy.stats import skew  # for dimensionality reduction
from glob import glob
import math

from tensorflow.keras.utils import Sequence

import logging


class DataIterator(Sequence):
    def __init__(self, base_path, sample_nrs=None, illum_nrs=24,
                 batch_size=2, patch_shape=(512, 512), image_shape=(512, 512), channels=1,
                 train_mode=False, quiet=False, dim_reduction=None, preserve_illumination_channels=True,
                 aug_rotation=None, aug_shift_bright=False, keras_iterator=True, epoch_multiplier=1):
        """
        Data generator for multi-illumination image stacks.
        It loads all the samples specified in sample_nrs from the base_path to the memory
        It specifies augmentation options applied in each iteration.

        :param base_path: Path to the dataset
        :param sample_nrs: List of sample numbers to be loaded (e.g. ['defect_001', '026', 'defect_002'])
        :param illum_nrs: List of image suffixes specifying the order of images in the stack
               (e.g ['101','201','102','202',...,'112',212']) or an integer (multiply of 12 & >12)
               or None if all 108 illuminations should be used
        :param batch_size: Number of samples in a batch returned by the generator (e.g. 2)
        :param patch_shape: Shape of patch returned by the generator. If smaller than image_shape a crop is returned.
               (e.g. (512, 512))
        :param image_shape: Shape of source images in the dataset (e.g. (512, 512))
        :param channels: Either 1 or 3 based on whether to load images as grayscale or as RGB
        :param train_mode: If true, randomly crops and augments the images, If false, all crops are selected
        :param quiet:
        :param dim_reduction: None - no reduction, subset of "albedo,normals" for Lambertian reduction, 
                              subset of "mean,std,skew" for statistical moments reduction
        :param preserve_illumination_channels: If True and aug_rotation equals 'rot30', 'rot60', 'rot90', or 'rot180'
                   it reshuffles the channels so that the illumination direction after the rotation
                   remains the same in each channel.
                   illum_nrs has to be ordered as [elev0_azim0, elev1_azim0, ... ,elev0_azim1, elev1_azim1, ...],
                   where elev is elevation and azim the azimuth of the illuminatior used while cappturing the image
                   aug_rotation must correspond to the angle between two adjacent illuminators on the same elevation.
        :param aug_rotation: Augment data by rotation: None - no rotations, 'rotRND' - random rotation,
                             'rot30','rot60','rot90', or 'rot180' - rotation by integer multiples of the angle selected
        :param aug_shift_bright: Augment data by shifting images by up to +-4px, changing the brightness by up to [+-5]
                                 for uint8 images or by +-0.02 if some dimensionality reduction is applied
        :param keras_iterator: If True, the iterator returns patches, masks, segmentations
                               If False, it also returns whether it has annotations, and locations
        :param epoch_multiplier: In training mode multiplies the length of an epoch
        """

        # Settings for loading the images
        self.channels = channels
        self.image_shape = image_shape
        self.sample_nrs = sample_nrs if sample_nrs is not None else os.listdir(base_path)
        self.total_samples = len(self.sample_nrs)

        all_allowed_illums = [str((i // 12 + 1) * 100 + (i % 12 + 1)) for i in range(108)]

        if illum_nrs is None:
            self.illum_nrs = np.array([str((i % 9 + 1) * 100 + (i // 9 + 1)) for i in range(108)])
        elif isinstance(illum_nrs, int) and illum_nrs >= 12 and illum_nrs % 12 == 0:
            divf = illum_nrs // 12
            self.illum_nrs = np.array([str((i % divf + 1) * 100 + (i // divf + 1)) for i in range(illum_nrs)])
        elif hasattr(illum_nrs, "__len__"):
            self.illum_nrs = np.array(illum_nrs)
        else:
            raise ValueError("illum_nrs must be either None, integer "
                             "(multiple of 12 and >12) or list of illuminations e.g. [101, 102, ...]")

        self.illum_ids = [all_allowed_illums.index(i) for i in self.illum_nrs]

        self.total_illums = len(self.illum_nrs)
        
        self.dim_reduction = dim_reduction
        self.lambertian_reduction = (dim_reduction is not None) and \
                                    (("normals" in dim_reduction) or ("albedo" in dim_reduction))
        self.moments_reduction = (dim_reduction is not None) and \
                                 (("mean" in dim_reduction) or ("std" in dim_reduction) or ("skew" in dim_reduction))

        self.sample_channels = self._get_sample_channels()
        self.patch_shape = patch_shape
        self.masks_ch_ids = [self.total_illums, self.total_illums + 1]

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
        self.preserve_illumination_channels = preserve_illumination_channels
        self.aug_rotation = aug_rotation
        self.aug_shift_bright = aug_shift_bright

        self.augment = (not self.aug_shift_bright) \
                       and (self.aug_rotation is None) \
                       and (self.patch_shape == self.image_shape)

        if not ((aug_rotation == 'rot30') or (aug_rotation == 'rot60') or (aug_rotation == 'rot90')
                or (aug_rotation == 'rot180') or (aug_rotation == 'rotRND') or (aug_rotation is None)):
            raise ValueError("aug_rotation has to be 'rot30','rot60','rot90', 'rot180', 'rotRND', or None")

        if (self.aug_rotation == "rot30" or self.aug_rotation == "rot60" or self.aug_rotation == "rot90" or
            self.aug_rotation == "rot180") and self.preserve_illumination_channels and self.dim_reduction:
            raise ValueError("Illumination preserving rotations cannot be"
                             " combined with dimensionality reduction techniques")

        if self.aug_rotation is not None and self.aug_rotation != "rot30" and self.aug_rotation != "rot60" and \
           self.aug_rotation != "rot90" and self.aug_rotation != "rot180" and self.preserve_illumination_channels:
            raise ValueError("Preserving channel ordering after augmentation is possible only when using "
                             "'rot30', 'rot60', 'rot90', or 'rot180' rotations")

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
            if self.preserve_illumination_channels:
                logging.warning("Illumination-preserving rotations are"
                                " not available for {} rotation.".format(self.aug_rotation))
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
            logging.info("Views to consider:", self.illum_nrs)
            if not train_mode:
                logging.info("Test_locations: %d" % len(self.locations))

            if self.augment:
                logging.info("Augmentations: None")
            else:
                logging.info("Augmentations:")
                logging.info("  Rotation:", self.aug_rotation)
                logging.info("  Channel order preserved if possible", self.preserve_illumination_channels)
                logging.info("  Brightness+Shifts:", self.aug_shift_bright)

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
        images = patches[..., :(self.channels * self.sample_channels)].astype(np.float32)
        segmentations = patches[..., -1].astype(np.float32)[..., None]
        masks = patches[..., -2].astype(np.float32)[..., None]

        if not self.dim_reduction:
            return images / 255., segmentations / 255., masks / 255.
        else:
            return images, segmentations, masks

    def _get_sample_channels(self):
        """
        Gets the number of channels for sample based on the number of illuminations,
         channels, and dimensionality reduction technique
        :return: Number of channels per sample (excluding the masks)
        """
        if self.dim_reduction is None:
            return self.total_illums * self.channels
        total_nr_channels = 0
        if "mean" in self.dim_reduction:
            total_nr_channels += 1
        if "std" in self.dim_reduction:
            total_nr_channels += 1
        if "skew" in self.dim_reduction:
            total_nr_channels += 1
        if "normals" in self.dim_reduction:
            total_nr_channels += 3
        if "albedo" in self.dim_reduction:
            total_nr_channels += 1
        return total_nr_channels
    
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
        data_dtype = np.uint8 if self.dim_reduction is None else np.float32
        mask_channels = 2
        # In case of rotation by 30 or 60, preload the rotated versions to memory
        total_nr_samples = self.total_samples if self.prerotate_angle == 0 else self.total_samples*3

        data = np.zeros((total_nr_samples,) + self.image_shape + (self.sample_channels + mask_channels,), dtype=data_dtype)
        has_annotations = np.zeros((self.total_samples,), dtype=np.bool)

        if self.lambertian_reduction:
            light_currents = np.genfromtxt(path.join(base_path, '..', 'light_currents.csv'), delimiter=',')[self.illum_ids]
            light_intensities = light_currents / np.max(light_currents)
            light_vectors = np.genfromtxt(path.join(base_path, '..', 'light_vectors.csv'), delimiter=',')[self.illum_ids]
            coeff = []
            LNY = []
            invYLYLT = []
            for i in range(1 if self.prerotate_angle == 0 else 3):
                L = self._rotate_vectors(light_vectors, self.prerotate_angle*i)
                # for albedo estimation
                LN = L.dot([0, 0, 1])
                LNY.append(LN * light_intensities)
                coeff.append(1. / np.dot(LNY[i], LNY[i]))
                # for normals estimation
                YL = L * light_intensities[:, None]
                invYL = np.linalg.inv(np.matmul(YL.T, YL))
                invYLYLT.append(np.matmul(invYL, YL.T))
        else:
            coeff = None
            LNY = None
            invYLYLT = None

        raw_sample_channels = self.channels * self.total_illums + mask_channels
        # For each sample
        for j, sample_dir in enumerate(self.sample_nrs):
            seg_filename = glob(path.join(base_path, sample_dir) + "/*_segmentation.png")[0]
            prefix = path.basename(seg_filename).split("_")[0]
            mask_filename = path.join(base_path, sample_dir, prefix + "_" + sample_dir + "_mask.png")
            
            # Load foreground segmentation mask
            with Image.open(seg_filename) as seg:
                sampledata = np.zeros(seg.size[::-1] + (raw_sample_channels,), dtype=np.uint8)
                if seg.mode == "LA":
                    sampledata[..., -1] = np.array(seg, dtype=np.uint8)[..., 0]
                else:
                    sampledata[..., -1] = np.array(seg, dtype=np.uint8)

            # Load defect segmentation mask
            if mask_filename is not None and path.isfile(mask_filename):
                with Image.open(mask_filename) as maskf:
                    if maskf.mode == "LA":
                        sampledata[..., -2] = np.array(maskf, dtype=np.uint8)[..., 0]
                    else:
                        sampledata[..., -2] = np.array(maskf, dtype=np.uint8)

                has_annotations[j] = True

            # Load stack of images
            for i, illum_nr in enumerate(self.illum_nrs):
                image_filename = path.join(base_path, sample_dir, prefix + "_" + sample_dir + "_" + illum_nr + ".png")
                with Image.open(image_filename) as image:
                    if self.channels == 1 and image.mode in ["RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"]:
                        image = image.convert(mode="L")
                        image = np.array(image, dtype=np.uint8)[..., np.newaxis]
                    else:
                        image = np.array(image, dtype=np.uint8)[..., self.channels]

                    sampledata[..., i*self.channels:(i+1)*self.channels] = image

            if self.dim_reduction is None:
                for k in range(1 if self.prerotate_angle == 0 else 3):
                    rot_sampledata = self.rotate(sampledata, k*self.prerotate_angle, interp="bicubic")
                    if self.preserve_illumination_channels and k > 0:
                        channel_shift = self.sample_channels // (360 // self.prerotate_angle)
                        ids = np.concatenate((np.roll(np.arange(self.sample_channels), k * channel_shift), self.masks_ch_ids))
                        rot_sampledata = rot_sampledata[..., ids]
                    data[j + k*self.total_samples] = rot_sampledata
            elif self.moments_reduction:
                for k in range(1 if self.prerotate_angle == 0 else 3):
                    data[j + k*self.total_samples] = self._reduce_moments(self.rotate(sampledata, k*self.prerotate_angle, interp="bicubic"))
            elif self.lambertian_reduction:
                for k in range(1 if self.prerotate_angle == 0 else 3):
                    data[j + k*self.total_samples] = self._reduce_lambertian(self.rotate(sampledata, k*self.prerotate_angle, interp="bicubic"), coeff[0], LNY[0], invYLYLT[0])
            else:
                raise ValueError("Unknown dimensionality reduction")

        return data, has_annotations
    
    def _reduce_moments(self, in_array):
        """
        Given an input array it computes the moemnts along the first self.total_illums channels, 
        it stacks these moments along the channel dimension and keeps the rest of the channels unchanged
        :param in_array: Input array
        :return: Array with reduced number of channels
        """
        featuresstack = []
        feats = in_array[..., :self.total_illums] / 255.
        if "mean" in self.dim_reduction:
            featuresstack.append(feats.mean(axis=-1, keepdims=True))
        if "std" in self.dim_reduction:
            featuresstack.append(feats.std(axis=-1, keepdims=True))
        if "skew" in self.dim_reduction:
            featuresstack.append(skew(feats, axis=-1)[..., np.newaxis])
        featuresstack.append(in_array[..., self.total_illums:] / 255.)
        return np.concatenate(featuresstack, axis=-1)
        
    def _reduce_lambertian(self, in_array, coef, LNY, invYLYLT):
        """
        Given an input array it computes the lambertain normals or/and albedo from the first self.total_illums channels, 
        it stacks these quantities along the channel dimension and keeps the rest of the channels unchanged
        :param in_array: Input array
        :param coef: Precomputed coeff for computing albedo
        :param LNY: Precomputed dot(light_direction, normal_driection)*light_intensity
        :param invYLYLT: Precomputed matmul(inv(np.matmul(YL.T, YL)), YL.T), where YL = L * light_intensities
        :return: Array with reduced number of channels
        """
        shape = in_array.shape
        images = np.reshape(in_array[..., :self.total_illums], (shape[0]*shape[1], -1)).T
        if "albedo" in self.dim_reduction and "normals" in self.dim_reduction:
            N = np.matmul(invYLYLT, images / 255.)
            albedo = np.linalg.norm(N, axis=0)[None, ...]
            np.divide(N, albedo, out=N, where=albedo > 0)
            N = np.concatenate((N, albedo), axis=0)
            normals = N.T.reshape((shape[0], shape[1], -1))
            return np.concatenate((normals, (in_array[..., self.total_illums:] / 255.)), axis=-1)
        elif "normals" in self.dim_reduction:
            N = np.matmul(invYLYLT, images / 255.)
            normals = N.T.reshape((shape[0], shape[1], -1))
            return np.concatenate((normals, (in_array[..., self.total_illums:] / 255.)), axis=-1)
        elif "albedo" in self.dim_reduction:
            albedo = coef * np.matmul(LNY, images / 255.).reshape((shape[0], shape[1], 1))
            return np.concatenate((albedo, (in_array[..., self.total_illums:] / 255.)), axis=-1)

    def augment_fn(self, in_array, do_shift, do_brightness,
                   illum_pres_rot_angle, do_arb_rotation, bb):
        in_array = np.copy(in_array)

        if illum_pres_rot_angle != 0:
            if illum_pres_rot_angle == 90:
                k_roll = np.random.randint(4)
                k_rot = k_roll
            elif illum_pres_rot_angle == 180:
                k_roll = np.random.randint(2)
                k_rot = k_roll * 2
            else:
                raise ValueError("Only 90 and 180 degrees are supported")

            if self.preserve_illumination_channels:
                channel_shift = self.sample_channels // (360 // illum_pres_rot_angle)
                ids = np.roll(np.arange(self.sample_channels), k_roll * channel_shift)
                ids = np.concatenate((ids, self.masks_ch_ids))
                in_array = np.rot90(in_array, k=k_rot)[..., ids]
            else:
                in_array = np.rot90(in_array, k=k_rot)

            if self.lambertian_reduction and "normals" in self.dim_reduction:
                rad_rot = -np.deg2rad(90 * k_rot)
                rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                         (np.sin(rad_rot), np.cos(rad_rot))], axis=0)  # [x, y]

                in_array[..., :2] = np.matmul(in_array[..., :2], rot_matrices)

        if do_brightness:
            if self.dim_reduction is not None:
                brightness_shift = np.float32(np.random.uniform(-(5/255), 5/255))
                in_array[..., :self.sample_channels] += brightness_shift
            else:
                brightness_shift = np.random.randint(-5, 6, dtype=np.int16)
                imgs = np.cast[np.int16](in_array[..., :self.sample_channels]) + brightness_shift
                in_array[..., :self.sample_channels] = np.clip(imgs, 0, 255).astype(np.uint8)

        if do_arb_rotation:
            angle = np.random.randint(-359, 360)
            if angle < 0:
                in_array = np.fliplr(in_array)
            in_array = self.rotate(in_array, np.abs(angle))

            if self.lambertian_reduction and "normals" in self.dim_reduction:
                rad_rot = -np.deg2rad(np.abs(angle))
                rot_matrices = np.stack([(np.cos(rad_rot), -np.sin(rad_rot)),
                                         (np.sin(rad_rot), np.cos(rad_rot))], axis=0)  # [x, y]

                in_array[..., :2] = np.matmul(in_array[..., :2], rot_matrices)

                if angle < 0:
                    in_array = np.concatenate((-in_array[..., 0:1], in_array[..., 1:]), axis=-1)

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
                                               do_shift=self.aug_shift_bright,
                                               do_brightness=self.aug_shift_bright,
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

        if in_array.dtype == np.bool:
            in_array = np.cast[np.uint8](in_array)

        if len(in_array.shape) > 3:
            in_array = in_array.reshape(in_array.shape[:2] + (-1,))
        elif len(in_array.shape) == 2:
            in_array = in_array[..., np.newaxis]

        out = np.empty(in_array.shape, dtype=in_array.dtype)
        for i in range(in_array.shape[-1]):
            out[..., i] = cv2.warpAffine(in_array[..., i], rot_matrix, (cols, rows), flags=flag)

        out = out.reshape(former_shape)

        if former_type == np.bool:
            return out > 0.5
        else:
            return out

    def get_name(self):
        DR = ""
        if self.dim_reduction is not None:
            if "mean" in self.dim_reduction:
                DR += "mu"
            if "std" in self.dim_reduction:
                DR += "std"
            if "skew" in self.dim_reduction:
                DR += "skw"
            if "albedo" in self.dim_reduction:
                DR += "alb"
            if "normals" in self.dim_reduction:
                DR += "nml"
            if DR != "":
                DR = "-DimR" + DR

        no_illum_pres_rot = ""
        if (self.aug_rotation == "rot30" or self.aug_rotation == "rot60" or self.aug_rotation == "rot90" or
            self.aug_rotation == "rot180") and (not self.preserve_illumination_channels):
            no_illum_pres_rot = "-noillumpres"

        patch_shape = ""
        if self.patch_shape != (512, 512):
            patch_shape = "P" + str(self.patch_shape[0])

        return "{patch}K{rot}{no_illum_pres}-I{illums}{DR}".format(
            patch=patch_shape,
            rot=self.aug_rotation,
            no_illum_pres=no_illum_pres_rot,
            illums=len(self.illum_nrs),
            DR=DR
        )

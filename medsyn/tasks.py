from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
import functools
import itertools
import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform,distance_transform_edt
from numpy.linalg import norm
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from .labelling import FlippedGaussianLabeller,AnomalyLabeller
from .task_shape import EitherDeformedHypershapePatchMaker,PerlinPatchMaker
from .utils import *

def cut_paste(sample: npt.NDArray[float],
              source_to_blend: npt.NDArray[float],
              anomaly_corner: npt.NDArray[int],
              anomaly_mask: npt.NDArray[bool]) -> npt.NDArray[float]:

    repeated_mask = np.broadcast_to(anomaly_mask, source_to_blend.shape)

    sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))

    aug_sample = sample.copy()
    aug_sample[sample_slices][repeated_mask] = source_to_blend[repeated_mask]

    return aug_sample



class BaseTask(ABC):
    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        self.sample_labeller = sample_labeller
        self.rng = np.random.default_rng()

        self.min_anom_prop=0.3
        self.max_anom_prop=0.8

        self.anomaly_shape_maker = EitherDeformedHypershapePatchMaker(nsa_sample_dimension)
        self.all_kwargs = all_kwargs


    def apply(self,
              sample: npt.NDArray[float],
              *args, **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :return: sample with task applied and label map.
        """

        aug_sample = sample.copy()

        sample_shape = np.array(sample.shape[1:])
        anomaly_mask = np.zeros(sample_shape, dtype=bool)

        min_anom_prop = self.min_anom_prop
        max_anom_prop = self.max_anom_prop

        min_dim_lens = (min_anom_prop * sample_shape).round().astype(int)
        max_dim_lens = (max_anom_prop * sample_shape).round().astype(int)
        # print(min_dim_lens,max_dim_lens) [15,15],[205,205]

        dim_bounds = list(zip(min_dim_lens, max_dim_lens)) #[(15, 205), (15, 205)]

        # For random number of times
        sample_mask = None

        for i in range(2):

            # Compute anomaly mask
            curr_anomaly_mask, intersect_fn = self.anomaly_shape_maker.get_patch_mask_and_intersect_fn(dim_bounds,
                                                                                                       sample_shape)

            # Choose anomaly location
            anomaly_corner = self.find_valid_anomaly_location(curr_anomaly_mask, sample_mask, sample_shape)

            # Apply self-supervised task

            aug_sample = self.augment_sample(aug_sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn)

            anomaly_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)] |= curr_anomaly_mask

            # Randomly brake at end of loop, ensuring we get at least 1 anomaly
            if self.rng.random() > 0.5:
                break

        if self.sample_labeller is not None:
            return aug_sample, self.sample_labeller(aug_sample, sample, anomaly_mask)
        else:
            # If no labeller is provided, we are probably in a calibration process
            return aug_sample, np.expand_dims(anomaly_mask, 0)


    def find_valid_anomaly_location(self,
                                    curr_anomaly_mask: npt.NDArray[bool],
                                    sample_mask: Optional[npt.NDArray[bool]],
                                    sample_shape: npt.NDArray[int]):

        curr_anomaly_shape = np.array(curr_anomaly_mask.shape)
        min_corner = np.zeros(len(sample_shape))
        max_corner = sample_shape - curr_anomaly_shape

        # - Apply anomaly at location
        while True:
            anomaly_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

            # If the sample mask is None, any location within the bounds is valid
            if sample_mask is None:
                break
            # Otherwise, we need to check that the intersection of the anomaly mask and the sample mask is at least 50%
            target_patch_obj_mask = sample_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)]
            if (np.sum(target_patch_obj_mask & curr_anomaly_mask) / np.sum(curr_anomaly_mask)) >= 0.5:
                break

        return anomaly_corner


    def __call__(self,
                 sample: npt.NDArray[float],
                 *args,
                 **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :param **kwargs:
            * *sample_path*: Path to source image
        :return: sample with task applied and label map.
        """
        if len(sample.shape)==2:
            sample = np.expand_dims(sample,axis=0)

        aug_sample, aug_mask = self.apply(sample, *args, **kwargs)

        if len(aug_sample.shape)==3 and aug_sample.shape[0]==1:
            aug_sample = aug_sample.squeeze(0)

        if len(aug_mask.shape)==3 and aug_mask.shape[0]==1:
            aug_mask = aug_mask.squeeze(0)

        return aug_sample,aug_mask.astype(np.float)


    @abstractmethod
    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        """
        Apply self-supervised task to region at anomaly_corner covered by anomaly_mask
        :param sample: Sample to be augmented.
        :param sample_mask: Object mask of sample.
        :param anomaly_corner: Index of anomaly corner.
        :param anomaly_mask: Mask
        :param anomaly_intersect_fn: Function which, given a line's origin and direction, finds its intersection with
        the edge of the anomaly mask
        :return:
        """


class BasePatchBlendingTask(BaseTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller],
                 source_samples: list,
                 blend_images: Callable[[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], npt.NDArray[bool]],
                                        npt.NDArray[float]],

                 **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)
        self.source_samples = source_samples
        self.blend_images = blend_images


    def augment_sample(self,
                       sample: npt.NDArray[float], # aug sample
                       sample_mask: Optional[npt.NDArray[bool]], # None
                       anomaly_corner: npt.NDArray[int], # center
                       anomaly_mask: npt.NDArray[bool], # small anomaly mask
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_channels = sample.shape[0] # 1
        num_dims = len(sample.shape[1:]) #2

        # Sample source to blend into current sample
        source_sample = random.choice(self.source_samples)

        source_sample_shape = np.array(source_sample.shape[1:]) #(256,256)


        assert len(source_sample_shape) == num_dims, 'Source and target have different number of spatial dimensions: ' \
                                                     f's-{len(source_sample_shape)}, t-{num_dims}'

        assert source_sample.shape[0] == num_channels, \
            f'Source and target have different number of channels: s-{source_sample.shape[0]}, t-{num_channels}'

        # Compute INVERSE transformation matrix for parameters (rotation, resizing)
        # This is the backwards operation (final source region -> initial source region).

        trans_matrix = functools.reduce(lambda m, ds: accumulate_rotation(m,
                                                                          self.rng.uniform(-np.pi / 4, np.pi / 4),
                                                                          ds),
                                        itertools.combinations(range(num_dims), 2),
                                        np.identity(num_dims))

        # Compute effect on corner coords
        target_anomaly_shape = np.array(anomaly_mask.shape)

        corner_coords = np.array(np.meshgrid(*np.stack([np.zeros(num_dims), target_anomaly_shape], axis=-1),
                                             indexing='ij')).reshape(num_dims, 2 ** num_dims)

        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        init_grid_shape = max_trans_coords - min_trans_coords

        # Sample scale and clip so that source region isn't too big
        max_scale = np.min(0.8 * source_sample_shape / init_grid_shape)

        # Compute final transformation matrix
        scale_change = 1 + self.rng.exponential(scale=0.1)
        scale_raw = self.rng.choice([scale_change, 1 / scale_change])
        scale = np.minimum(scale_raw, max_scale)

        trans_matrix = accumulate_scaling(trans_matrix, scale)

        # Recompute effect on corner coord
        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        final_init_grid_shape = max_trans_coords - min_trans_coords

        # Choose anomaly source location
        final_init_grid_shape = final_init_grid_shape.astype(int)
        min_corner = np.zeros(len(source_sample_shape))
        max_corner = source_sample_shape - final_init_grid_shape

        source_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

        # Extract source
        source_orig = source_sample[get_patch_image_slices(source_corner, tuple(final_init_grid_shape))]


        # Because we computed the backwards transformation we don't need to inverse the matrix
        source_to_blend = np.stack([affine_transform(chan, trans_matrix, offset=-min_trans_coords,
                                                     output_shape=tuple(target_anomaly_shape))
                                    for chan in source_orig])

        spatial_axis = tuple(range(1, len(source_sample.shape)))
        # Spline interpolation can make values fall outside domain, so clip to the original range
        source_to_blend = np.clip(source_to_blend,
                                  source_sample.min(axis=spatial_axis, keepdims=True),
                                  source_sample.max(axis=spatial_axis, keepdims=True))


        # As the blending can alter areas outside the mask, update the mask with any effected areas

        aug_sample = self.blend_images(sample, source_to_blend, anomaly_corner, anomaly_mask)

        sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))
        sample_diff = np.mean(np.abs(sample[sample_slices] - aug_sample[sample_slices]), axis=0)

        anomaly_mask[sample_diff > 0.001] = True
        # Return sample with source blended into it
        return aug_sample



class BaseDeformationTask(BaseTask):

    @abstractmethod
    def compute_mapping(self,
                        sample: npt.NDArray[float],
                        sample_mask: Optional[npt.NDArray[bool]],
                        anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                        anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        """
        Returns array of size (*anomaly_mask.shape, len(anomaly_mask.shape)).
        Probably don't need entire sample, but including in for generality.
        :param sample:
        :param sample_mask:
        :param anomaly_corner:
        :param anomaly_mask:
        :param anomaly_intersect_fn:
        :return:
        """

    def augment_sample(self,
                       sample: npt.NDArray[float],
                       sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int],
                       anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_channels = sample.shape[0]
        mapping = self.compute_mapping(sample, sample_mask, anomaly_corner, anomaly_mask, anomaly_intersect_fn)
        sample_slices = get_patch_slices(anomaly_corner, tuple(anomaly_mask.shape))

        for chan in range(num_channels):
            sample[chan][sample_slices] = ndimage.map_coordinates(sample[chan][sample_slices],
                                                                  mapping,
                                                                  mode='nearest')
        return sample



class RadialDeformationTask(BaseDeformationTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 deform_factor: Optional[float] = None,
                 deform_centre: Optional[npt.NDArray] = None, **kwargs):

        super().__init__(sample_labeller, **kwargs)
        self.deform_factor = deform_factor
        self.deform_centre = deform_centre
        self.max_anom_prop = 0.6
        self.min_anom_prop = 0.2

    def get_deform_factor(self, def_centre: npt.NDArray[int], anomaly_mask: npt.NDArray[bool]):
        return self.deform_factor if self.deform_factor is not None else 2 ** self.rng.uniform(0.5, 2)

    @abstractmethod
    def compute_new_distance(self, curr_distance: float, max_distance: float, factor: float) -> float:
        """
        Compute new distance for point to be sampled from
        :param curr_distance:
        :param max_distance:
        :param factor:
        """

    def compute_mapping(self,
                        sample: npt.NDArray[float],
                        sample_mask: Optional[npt.NDArray[bool]],
                        anomaly_corner: npt.NDArray[int],
                        anomaly_mask: npt.NDArray[bool],
                        anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        # NOTE: This assumes that the shape is convex, will make discontinuities if it's not.

        anomaly_shape = np.array(anomaly_mask.shape)
        num_dims = len(anomaly_shape)

        # Expand so can later be broadcast with (D, N)
        mask_centre = (anomaly_shape - 1) / 2
        exp_mask_centre = np.reshape(mask_centre, (-1, 1))
        # Shape (D, N)
        poss_centre_coords = np.stack(np.nonzero(anomaly_mask))
        def_centre = self.deform_centre if self.deform_centre is not None else \
            poss_centre_coords[:, np.random.randint(poss_centre_coords.shape[1])]

        assert anomaly_mask[tuple(def_centre.round().astype(int))], f'Centre is not within anomaly: {def_centre}'

        # exp_ = expanded
        exp_def_centre = np.reshape(def_centre, (-1, 1))

        # (D, *anomaly_shape)
        mapping = np.stack(np.meshgrid(*[np.arange(s, dtype=float) for s in anomaly_shape], indexing='ij'), axis=0)

        # Ignore pixels on edge of bounding box
        mask_inner_slice = tuple([slice(1, -1)] * num_dims)
        map_inner_slice = tuple([slice(None)] + list(mask_inner_slice))
        # Get all coords and transpose so coord index is last dimension (D, N)
        anomaly_coords = mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])]

        all_coords_to_centre = anomaly_coords - exp_def_centre
        all_coords_distance = norm(all_coords_to_centre, axis=0)
        # Ignore zero divided by zero, as we correct it before mapping is returned
        with np.errstate(invalid='ignore'):
            all_coords_norm_dirs = all_coords_to_centre / all_coords_distance

        mask_edge_intersections = anomaly_intersect_fn(exp_def_centre - exp_mask_centre, all_coords_norm_dirs) + exp_mask_centre

        mask_edge_distances = norm(mask_edge_intersections - exp_def_centre, axis=0)

        # Get factor once, so is same for all pixels
        def_factor = self.get_deform_factor(def_centre, anomaly_mask)
        new_coord_distances = self.compute_new_distance(all_coords_distance, mask_edge_distances, def_factor)
        # (D, N)
        new_coords = exp_def_centre + new_coord_distances * all_coords_norm_dirs

        mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])] = new_coords

        # Revert centre coordinate, as it will be nan due to the zero magnitude direction vector
        mapping[(slice(None), *def_centre)] = def_centre
        return mapping



class CutPastePatchBlender(BasePatchBlendingTask):

    def __init__(self,
                 source_images: list,
                 Labelber_std: float= 0.2,
                 **kwargs):
        sample_labeller = FlippedGaussianLabeller(Labelber_std)
        source_images=[ np.expand_dims(image,axis=0) if len(image.shape)==2 else image for image in source_images]
        super().__init__(sample_labeller, source_images, cut_paste)
        self.max_anom_prop = 0.6
        self.min_anom_prop = 0.1
        self.anomaly_shape_maker = PerlinPatchMaker()



class SmoothIntensityChangeTask(BaseTask):

    def __init__(self,
                 intensity_task_scale: float,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)
        self.intensity_task_scale = intensity_task_scale
        self.max_anom_prop = 0.8
        self.min_anom_prop = 0.3
        self.anomaly_shape_maker = PerlinPatchMaker()

    def augment_sample(self,
                       sample: npt.NDArray[float], # aug_sample
                       sample_mask: Optional[npt.NDArray[bool]],# None
                       anomaly_corner: npt.NDArray[int], # anomaly center
                       anomaly_mask: npt.NDArray[bool], # small anomaly mask
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_chans = sample.shape[0] # 1
        sample_shape = sample.shape[1:] #(256,256)
        num_dims = len(sample_shape) # 2

        dist_map = distance_transform_edt(anomaly_mask)
        min_shape_dim = np.min(sample_shape) # 256

        smooth_dist = np.minimum(min_shape_dim * (0.02 + np.random.gamma(3, 0.01)), np.max(dist_map))
        smooth_dist_map = dist_map / smooth_dist
        smooth_dist_map[smooth_dist_map > 1] = 1
        # smooth_dist_map = 1

        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask.shape)

        # anomaly_pixel_stds = np.array([np.std(c[anomaly_mask]) for c in sample[anomaly_patch_slices]])
        # Randomly negate, so some intensity changes are subtractions

        intensity_changes = (self.intensity_task_scale / 2 + np.random.gamma(3, self.intensity_task_scale)) \
            * np.random.choice([1, -1], size=num_chans)

        intensity_change_map = smooth_dist_map * np.reshape(intensity_changes, [-1] + [1] * num_dims)

        new_patch = sample[anomaly_patch_slices] + intensity_change_map

        spatial_axis = tuple(range(1, len(sample.shape)))

        sample[anomaly_patch_slices] = np.clip(new_patch,
                                               sample.min(axis=spatial_axis, keepdims=True),
                                               sample.max(axis=spatial_axis, keepdims=True))

        return sample



class GaussIntensityChangeTask(BaseTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)
        self.max_anom_prop = 0.8
        self.min_anom_prop = 0.3
        self.sigma_bs = [4, 7]
        self.positive_range = [0.4, 0.6]
        self.negative_range = [-0.6, -0.4]
        self.anomaly_shape_maker = PerlinPatchMaker()
    def get_predefined_texture(self,
                               mask_shape,
                               sigma_b,
                               positive_range=None,
                               negative_range=None,
                               ):

        assert (positive_range is not None) or (negative_range is not None)

        random_sample = np.random.randn(mask_shape[0], mask_shape[1])

        random_sample = (random_sample >= 0.0).astype(float)  # int type can't do Gaussian filter

        random_sample = gaussian_filter(random_sample, sigma_b)

        random_sample = (random_sample - np.min(random_sample)) / (np.max(random_sample) - np.min(random_sample))

        if np.random.uniform(0, 1) <= 0.5:
            u_0 = np.random.uniform(positive_range[0], positive_range[1])
        else:
            if negative_range is not None:
                u_0 = np.random.uniform(negative_range[0], negative_range[1])
            else:
                u_0 = np.random.uniform(-positive_range[1], -positive_range[0])

        Bj = np.clip(u_0 * random_sample, -1, 1)
        return Bj

    def create_texture(self,sizes):
        texture = self.get_predefined_texture(sizes,
                                         random.choice(self.sigma_bs),
                                         self.positive_range,
                                         self.negative_range)
        return texture


    def augment_sample(self,
                       sample: npt.NDArray[float], # aug_sample
                       sample_mask: Optional[npt.NDArray[bool]],# None
                       anomaly_corner: npt.NDArray[int], # anomaly center
                       anomaly_mask: npt.NDArray[bool], # small anomaly mask
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        anomaly_mask_copy = anomaly_mask.astype(np.float)
        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask_copy.shape)

        texture = self.create_texture(sample.shape[1:])

        while True:
            if len(texture.shape)<len(sample.shape):
                texture=np.expand_dims(texture,0)
            else:
                break

        sample = sample / 255.0
        sigma = np.random.uniform(1, 4)
        geo_blur = gaussian_filter(anomaly_mask_copy, sigma)

        for cha in range(sample.shape[0]):
            sample[anomaly_patch_slices][cha] = sample[anomaly_patch_slices][cha] * (1 - anomaly_mask_copy) + \
                                           (sample[anomaly_patch_slices][cha] + texture[anomaly_patch_slices][cha] * geo_blur) * anomaly_mask_copy

        sample = np.clip(sample, a_min=0, a_max=1.0)
        sample = sample * 255.0
        return sample



class IdentityTask(BaseTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)

    def augment_sample(self,
                       sample: npt.NDArray[float], # aug_sample
                       sample_mask: Optional[npt.NDArray[bool]],# None
                       anomaly_corner: npt.NDArray[int], # anomaly center
                       anomaly_mask: npt.NDArray[bool], # small anomaly mask
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        anomaly_mask[:,:] = False
        return sample



class SinkDeformationTask(RadialDeformationTask):
    # y = 1 - (1 - x)^3 (between 0 and 1)
    # -> y = max_d (1 - (1 - curr / max_d) ^ factor)
    # -> y = max_d - (max_d - curr) ^ factor / max_d ^ (factor - 1)

    def compute_new_distance(self, curr_distance: Union[float, npt.NDArray[float]],
                             max_distance: Union[float, npt.NDArray[float]],
                             factor: Union[float, npt.NDArray[float]]) -> Union[float, npt.NDArray[float]]:

        return max_distance - (max_distance - curr_distance) ** factor / max_distance ** (factor - 1)



class SourceDeformationTask(RadialDeformationTask):

    def compute_new_distance(self, curr_distance: Union[float, npt.NDArray[float]],
                             max_distance: Union[float, npt.NDArray[float]],
                             factor: Union[float, npt.NDArray[float]]) -> Union[float, npt.NDArray[float]]:
        # y = x^3 (between 0 and 1)
        # -> y = max_d * (curr / max) ^ factor
        # -> y = curr ^ factor / max_d ^ (factor - 1)   to avoid FP errors
        return curr_distance ** factor / max_distance ** (factor - 1)
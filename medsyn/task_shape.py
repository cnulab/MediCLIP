from abc import abstractmethod, ABC
import functools
import itertools
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
from skimage.morphology import convex_hull_image

from .utils import accumulate_rotation, accumulate_scaling
import math
import imgaug.augmenters as iaa


class PatchShapeMaker(ABC):

    @abstractmethod
    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:
        """
        :param dim_bounds: Tuples giving lower and upper bounds for patch size in each dimension
        :param img_dims: Image dimensions, can be used as scaling factor.
        Creates a patch mask to be used in the self-supervised task.
        Mask must have length(dim_bounds) dimensions.
        """
        pass

    def __call__(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:
        return self.get_patch_mask(dim_bounds, img_dims)


class PerlinPatchMaker(PatchShapeMaker):

    def __init__(self,
                 min_perlin_scale=0,
                 perlin_scale=1,
                 perlin_noise_threshold = 0.3,
                 perlin_min_size = 0.2
                 ):

        self.min_perlin_scale = min_perlin_scale
        self.perlin_scale = perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.perlin_min_size = perlin_min_size

    def lerp_np(self, x, y, w):
        fin_out = (y - x) * w + x
        return fin_out

    def rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

        tile_grads = lambda slice1, slice2: np.repeat(
            np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
        dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                         axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * self.lerp_np(self.lerp_np(n00, n10, t[..., 0]), self.lerp_np(n01, n11, t[..., 0]), t[..., 1])


    def get_patch_mask_and_intersect_fn(self,
                                        dim_bounds: List[Tuple[int, int]],
                                        img_dims: np.ndarray) \
            -> Tuple[npt.NDArray[bool], Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]]:

        perlin_scalex = 2 ** (np.random.randint(low = self.min_perlin_scale, high = self.perlin_scale, size=(1,))[0])
        perlin_scaley = 2 ** (np.random.randint(low = self.min_perlin_scale, high = self.perlin_scale, size=(1,))[0])

        noise_size_x = np.random.randint(low=dim_bounds[0][0],high=dim_bounds[0][1])
        noise_size_y = np.random.randint(low=dim_bounds[1][0],high=dim_bounds[1][1])

        while True:
            perlin_noise = self.rand_perlin_2d_np((noise_size_x, noise_size_y), (perlin_scalex, perlin_scaley))

            # apply affine transform
            rot = iaa.Affine(rotate=(-90, 90))
            perlin_noise = rot(image=perlin_noise)

            # make a mask by applying threshold
            mask_noise = np.where(
                perlin_noise > self.perlin_noise_threshold,
                np.ones_like(perlin_noise),
                np.zeros_like(perlin_noise)
            )
            mask_noise[mask_noise != 0] = 1.0

            if np.mean(mask_noise) >= self.perlin_min_size:
                break
        return mask_noise.astype(np.bool), None


    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> npt.NDArray[bool]:
        return self.get_patch_mask_and_intersect_fn(dim_bounds, img_dims)[0]



class DeformedHypershapePatchMaker(PatchShapeMaker):

    def __init__(self, aligned_distance_to_edge_fn: Callable[[npt.NDArray[int], npt.NDArray[float], npt.NDArray[float]],
                                                             npt.NDArray[float]],
                 sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):
        # Instead of mask function, need inclusion test function
        # takes shape + point as parameters
        self.aligned_distance_to_edge_fn = aligned_distance_to_edge_fn
        self.sample_dist = sample_dist
        self.rng = np.random.default_rng()

    @abstractmethod
    def within_aligned_shape(self, array_of_coords: npt.NDArray[float], shape_size: npt.NDArray[int]) \
            -> npt.NDArray[bool]:
        """
        Calculates whether a point is within an aligned shape with dimensions shape_size.
        :param array_of_coords:
        :param shape_size:
        """

    def make_shape_intersect_function(self,
                                      inv_trans_matrix: npt.NDArray[float],
                                      trans_matrix: npt.NDArray[float],
                                      shape_size: npt.NDArray[int]) \
            -> Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]:

        return lambda orig, direction: trans_matrix @ self.aligned_distance_to_edge_fn(shape_size,
                                                                                       inv_trans_matrix @ orig,
                                                                                       inv_trans_matrix @ direction)

    def get_patch_mask_and_intersect_fn(self,
                                        dim_bounds: List[Tuple[int, int]],
                                        img_dims: np.ndarray) \
            -> Tuple[npt.NDArray[bool], Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]]:

        shape = np.array([self.sample_dist(lb, ub, d) for ((lb, ub), d) in zip(dim_bounds, img_dims)])

        num_dims = len(dim_bounds)

        trans_matrix = np.identity(num_dims)

        # Instead of transforming mask, accumulate transformation matrix
        for d in range(num_dims):
            other_dim = self.rng.choice([i for i in range(num_dims) if i != d])
            shear_factor = self.rng.normal(scale=0.2)
            trans_matrix[d, other_dim] = shear_factor

        # Rotate mask, using all possible access combinations
        trans_matrix = functools.reduce(lambda m, ds: accumulate_rotation(m,
                                                                          self.rng.uniform(-np.pi / 2, np.pi / 2),
                                                                          ds),
                                        itertools.combinations(range(num_dims), 2),
                                        trans_matrix)

        # Using corner points, calculate size of resulting shape
        shape_width = (shape - 1) / 2
        corner_coord_grid = np.array(np.meshgrid(*np.stack([shape_width, -shape_width], axis=-1), indexing='ij'))
        corner_coords = corner_coord_grid.reshape(num_dims, 2 ** num_dims)

        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        final_grid_shape = max_trans_coords + 1 - min_trans_coords

        # Check if transformations have made patch too big
        ub_shape = np.array([ub for _, ub in dim_bounds])
        if np.any(final_grid_shape > ub_shape):
            # If so, scale down to be within limits.
            max_scale_diff = np.max(final_grid_shape / ub_shape)
            trans_matrix = accumulate_scaling(trans_matrix, 1 / max_scale_diff)

            # Repeat calculations with new transformation matrix
            trans_corner_coords = trans_matrix @ corner_coords
            min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
            max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
            final_grid_shape = max_trans_coords + 1 - min_trans_coords

        # Create meshgrid of coords of resulting shape
        coord_ranges = [np.arange(lb, ub + 1) for lb, ub in zip(min_trans_coords, max_trans_coords)]
        coord_grid = np.array(np.meshgrid(*coord_ranges, indexing='ij'))

        # Apply inverse transformation matrix, to compute sampling points
        inv_trans_matrix = np.linalg.inv(trans_matrix)
        inv_coord_grid_f = inv_trans_matrix @ np.reshape(coord_grid, (num_dims, -1))

        # Apply inclusion test function, giving an array containing a boolean for each coordinate.
        inv_result_grid_f = self.within_aligned_shape(inv_coord_grid_f, shape)

        return np.reshape(inv_result_grid_f, final_grid_shape.astype(int)), \
            self.make_shape_intersect_function(inv_trans_matrix, trans_matrix, shape)


    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> npt.NDArray[bool]:
        return self.get_patch_mask_and_intersect_fn(dim_bounds, img_dims)[0]





def intersect_to_aligned_hyperrectangle_edge(hyperrectangle_shape: npt.NDArray[int],
                                             origin: npt.NDArray[float],
                                             direction: npt.NDArray[float]) \
        -> npt.NDArray[float]:
    """

    :param hyperrectangle_shape: Numpy array of hyperrectangle shape, (D)
    :param origin: Numpy array of origin coordinates (D, N)
    :param direction: Numpy array of normalised direction vectors (D, N)
    :return:
    """
    rect_width = np.reshape(hyperrectangle_shape / 2, (-1, 1))
    # Normalise direction magnitudes
    direction = direction / np.linalg.norm(direction, axis=0)

    max_dir = rect_width - origin
    min_dir = - rect_width - origin

    # Shape (2D, N)
    all_edge_distances = np.concatenate([max_dir / direction,
                                         min_dir / direction])

    # Set any behind distances to inf, so we don't choose them
    all_edge_distances[all_edge_distances < 0] = np.inf

    # Shape (N,)
    min_distances = np.min(all_edge_distances, axis=0)

    min_is_inf = min_distances == np.inf
    assert not np.any(min_is_inf), f'Some lines are outside bounding box:\n' \
                                   f'rectangle shape: {hyperrectangle_shape},' \
                                   f'origins - {origin[min_is_inf]},\n' \
                                   f'directions -  {direction[min_is_inf]}'

    return origin + direction * min_distances



class DeformedHyperrectanglePatchMaker(DeformedHypershapePatchMaker):

    def __init__(self, sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):
        super().__init__(intersect_to_aligned_hyperrectangle_edge, sample_dist)

    def within_aligned_shape(self, array_of_coords: npt.NDArray[float], shape_size: npt.NDArray[int]) \
            -> npt.NDArray[bool]:
        rect_width = np.reshape(shape_size / 2, (-1, 1))
        return np.all(-rect_width <= array_of_coords, axis=0) & np.all(array_of_coords <= rect_width, axis=0)



def intersect_aligned_hyperellipse_edge(hyperellipse_shape: npt.NDArray[int],
                                        origin: npt.NDArray[float],
                                        direction: npt.NDArray[float]) \
        -> npt.NDArray[float]:
    ellipse_radii_sq = np.reshape((hyperellipse_shape / 2) ** 2, (-1, 1))
    # Normalise direction magnitudes
    direction = direction / np.linalg.norm(direction, axis=0)

    # Compute quadratic coefficients, all shape (N)
    a = np.sum(direction ** 2 / ellipse_radii_sq, axis=0)
    b = np.sum(2 * origin * direction / ellipse_radii_sq, axis=0)
    c = np.sum(origin ** 2 / ellipse_radii_sq, axis=0) - 1

    # Solve quadratic, (N)
    det = b ** 2 - 4 * a * c

    det_is_negative = det < 0
    assert not np.any(det_is_negative), f'Some lines never intersect ellipse:\n' \
                                        f'Ellipse shape: {hyperellipse_shape}\n' \
                                        f'origins: {origin[det_is_negative]}' \
                                        f'directions: {direction[det_is_negative]}'

    solutions = (-b + np.array([[1], [-1]]) * np.sqrt(det)) / (2 * a)

    # Make any negative solutions (behind origin) infinity so we don't choose them
    solutions[solutions < 0] = np.inf

    min_solutions = np.min(solutions, axis=0)
    min_is_inf = min_solutions == np.inf
    assert not np.any(min_is_inf), f'Some lines are outside ellipse:\n' \
                                   f'ellipse shape: {hyperellipse_shape},' \
                                   f'origins - {origin[min_is_inf]},\n' \
                                   f'directions -  {direction[min_is_inf]}'
    return origin + direction * min_solutions


class DeformedHyperellipsePatchMaker(DeformedHypershapePatchMaker):

    def __init__(self, sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):
        super().__init__(intersect_aligned_hyperellipse_edge, sample_dist)

    def within_aligned_shape(self, array_of_coords: npt.NDArray[float], shape_size: npt.NDArray[int]) \
            -> npt.NDArray[bool]:
        ellipse_radii = np.reshape(shape_size / 2, (-1, 1))
        return np.sum(array_of_coords ** 2 / ellipse_radii ** 2, axis=0) <= 1


class CombinedDeformedHypershapePatchMaker(PatchShapeMaker):

    def __init__(self, sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):

        self.rect_maker = DeformedHyperrectanglePatchMaker(sample_dist)
        self.ellip_maker = DeformedHyperellipsePatchMaker(sample_dist)
        self.last_choice = None

    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> npt.NDArray[bool]:

        mode = np.random.choice(['rect', 'ellipse', 'comb'])
        self.last_choice = mode

        if mode == 'rect':
            return self.rect_maker(dim_bounds, img_dims)

        elif mode == 'ellipse':
            return self.ellip_maker(dim_bounds, img_dims)

        elif mode == 'comb':

            rect_mask = self.rect_maker(dim_bounds, img_dims)
            ellip_mask = self.ellip_maker(dim_bounds, img_dims)

            rect_size = np.sum(rect_mask)
            ellip_size = np.sum(ellip_mask)

            big_m, small_m = (rect_mask, ellip_mask) if rect_size >= ellip_size else (ellip_mask, rect_mask)

            # Choose point on big mask to put small mask
            mask_coords = np.nonzero(big_m)

            point_ind = self.rect_maker.rng.integers(len(mask_coords[0]))

            chosen_coord = np.array([m_cs[point_ind] for m_cs in mask_coords])

            small_shape = np.array(small_m.shape)
            lower_coord = chosen_coord - small_shape // 2

            if np.any(lower_coord < 0):
                to_pad_below = np.maximum(-lower_coord, 0)
                big_m = np.pad(big_m, [(p, 0) for p in to_pad_below])
                lower_coord += to_pad_below

            big_shape = np.array(big_m.shape)

            upper_coord = lower_coord + small_shape
            if np.any(upper_coord > big_shape):
                to_pad_above = np.maximum(upper_coord - big_shape, 0)
                big_m = np.pad(big_m, [(0, p) for p in to_pad_above])

            big_m[tuple([slice(lb, ub) for lb, ub in zip(lower_coord, upper_coord)])] |= small_m

            return convex_hull_image(big_m)

        else:
            raise Exception('Invalid mask option')


class EitherDeformedHypershapePatchMaker(PatchShapeMaker):

    def __init__(self,
                 sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):

        self.rect_maker = DeformedHyperrectanglePatchMaker(sample_dist)
        self.ellip_maker = DeformedHyperellipsePatchMaker(sample_dist)

    def get_patch_mask_and_intersect_fn(self,
                                        dim_bounds: List[Tuple[int, int]],
                                        img_dims: np.ndarray) \
            -> Tuple[npt.NDArray[bool], Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]]:

        chosen_task = np.random.choice([self.rect_maker, self.ellip_maker])
        return chosen_task.get_patch_mask_and_intersect_fn(dim_bounds, img_dims)


    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> npt.NDArray[bool]:
        return self.get_patch_mask_and_intersect_fn(dim_bounds, img_dims)[0]

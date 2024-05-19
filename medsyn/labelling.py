from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import morphology


class AnomalyLabeller(ABC):
    @abstractmethod
    def label(self,  aug_img: npt.NDArray[float], orig_img: npt.NDArray[float], mask: npt.NDArray[bool]) \
            -> npt.NDArray[float]:
        """
        :param aug_img: Image with anomaly augmentation applied.
        :param orig_img: Original image, prior to anomalies.
        :param mask: Mask of where the image has been altered.
        """
        pass

    def __call__(self, aug_img: npt.NDArray[float], orig_img: npt.NDArray[float], mask: npt.NDArray[bool]) \
            -> npt.NDArray[float]:

        return self.label(aug_img, orig_img, mask)


class IntensityDiffLabeller(AnomalyLabeller, ABC):

    def __init__(self):
        super().__init__()
        self.binary_structures = {}

    @abstractmethod
    def label_fn(self, x: Union[npt.NDArray[float], float]) -> Union[npt.NDArray[float], float]:
        pass

    def label(self,
              aug_img: npt.NDArray[float],
              orig_img: npt.NDArray[float],
              mask: npt.NDArray[bool]) \
            -> npt.NDArray[float]:
        """
        :param aug_img: Image with patches blended within it.
        :param orig_img: Original image, prior to anomalies.
        :param mask: Mask of where the image has been altered.
        """

        avg_diff = np.mean(mask * np.abs(aug_img - orig_img), axis=0)

        scaled_diff = self.label_fn(avg_diff)

        assert np.all(scaled_diff >= 0)

        num_dims = len(mask.shape)
        if num_dims not in self.binary_structures:
            self.binary_structures[num_dims] = ndimage.generate_binary_structure(num_dims, 2)

        bin_structure = self.binary_structures[num_dims]

        for anom_slice in ndimage.find_objects(ndimage.label(mask)[0]):
            anom_region_label = ndimage.grey_closing(scaled_diff[anom_slice], footprint=bin_structure)

            recon_seed = np.copy(anom_region_label)
            recon_seed[num_dims * (slice(1, -1),)] = anom_region_label.max()
            scaled_diff[anom_slice] = morphology.reconstruction(recon_seed, anom_region_label,
                                                                method='erosion',
                                                                )
        return scaled_diff



class SaturatingLabeller(IntensityDiffLabeller):

    def __init__(self, a: float, c: float):
        """
        Labeller using transformed sigmoid function: (1 + c) / (1 + e^(-ax+b)) - c
        Function range is [-c, 1]
        """
        super().__init__()
        self.a = a
        self.c = c

    def label_fn(self, x: Union[npt.NDArray[float], float]) -> Union[npt.NDArray[float], float]:
        return (1 + self.c) / (1 + np.exp(-self.a * x) / self.c) - self.c

    def __call__(self, aug_img: npt.NDArray[float], orig_img: npt.NDArray[float], mask: npt.NDArray[bool]) \
            -> npt.NDArray[float]:
        return self.label(aug_img, orig_img, mask)


class FlippedGaussianLabeller(IntensityDiffLabeller):
    def __init__(self, std: float):
        super(FlippedGaussianLabeller, self).__init__()
        self.std = std

    def label_fn(self, x: Union[npt.NDArray[float], float]) -> Union[npt.NDArray[float], float]:
        return 1 - np.exp(-x**2 / (2 * self.std**2))

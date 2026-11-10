"""MRI forward model: centered FFTs, the SENSE operator, the undersampling masks and their application."""
import torch
import torch.nn as nn
import numpy as np
import contextlib
from typing import Optional, Sequence, Tuple, Union
from enum import Enum

def fftn(data, dim=None, norm="ortho", centered=True):
    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    if centered:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def ifftn(data, dim=None, norm="ortho", centered=True):
    if centered:
        data = torch.fft.fftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    return data


# ----------------------------------------------------------------------
# SENSE operator
# ----------------------------------------------------------------------

class SenseOperator(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation computes:
        y = (W * F * S) * x

    The adjoint operation computes:
        x = (S^H * F^H * W) y

    """

    def __init__(self, maps, mask=1.):
        '''
        Args:
            - maps (torch.Tensor[complex64]) : [nbatch, ncoils, nx, ny] 
                sensitivity maps
            - mask (torch.Tensor[bool]): [1, 1, ny, ny] 
                sampling maksk in k-space
        '''
        super(SenseOperator, self).__init__()
        self.maps = maps
        self.mask = mask
        nb, nc, nx, ny = maps.shape
        self.ishape = (nb, nc, nx, ny)
        self.oshape = (nb, 1, nx, ny)

    def forward(self, input, adjoint=False):

        if adjoint:
            ksp_coils = input * self.mask
            img_coils = ifftn(ksp_coils, dim=[-1,-2])
            output = torch.sum(img_coils * self.maps.conj(), dim=1)

        else:
            coils = input[:, None] * self.maps
            ksp_coils = fftn(coils, dim=[-1,-2])
            ksp_coils *= self.mask
            output = ksp_coils

        return output


# ----------------------------------------------------------------------
# Undersampling masks (modified from fastMRI)
# ----------------------------------------------------------------------

"""
Modified from the original file in the fastMRI repository:
https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/subsample.py
"""



class MaskType(Enum):
    ONE_D = 1
    TWO_D = 2


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float] = None,
        calib_sizes: Sequence[int] = None,
        accelerations: Sequence[int] = None,
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
        mask_type: MaskType=None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
            mask_type: Type of mask: 1D or 2D
        """
        if not calib_sizes and not center_fractions:
            raise ValueError("Either calib_sizes or center_fractions must be specified.")
        if calib_sizes and center_fractions:
            raise ValueError("Only one of calib_sizes or center_fractions can be specified")
        if center_fractions:
            if len(center_fractions) != len(accelerations) and not allow_any_combination:
                raise ValueError(
                    "Number of center fractions should match number of accelerations "
                    "if allow_any_combination is False."
                )
        if calib_sizes:
            if len(calib_sizes) != len(accelerations) and not allow_any_combination:
                raise ValueError(
                    "Length of calib_sizes should match number of accelerations "
                    "if allow_any_combination is False."
                )
        if type(mask_type) is not MaskType:
            raise ValueError(
                'mask_type must to be of type %s' % MaskType
            )

        self.center_fractions = center_fractions
        self.calib_sizes = calib_sizes
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.mask_type = mask_type

        self.rng = np.random.RandomState(seed)


    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space [N,H,W].
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines. Each element in the masks (and center frequency lines)
            contain a different mask that corresponds to a different element in the batch.

        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        masks = []
        num_low_freqs = []
        with temp_seed(self.rng, seed):
            for batch_idx in range(shape[0]):
                center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                    shape, offset
                )

            # combine masks together
            masks.append(torch.max(center_mask, accel_mask))
            num_low_freqs.append([num_low_frequencies])

        return torch.cat(masks), torch.tensor(num_low_freqs)


    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_h = shape[1]
        num_w = shape[2]
        if self.center_fractions:
            center_fraction, acceleration = self.choose_acceleration()
            num_low_frequencies = round(num_h * center_fraction)
        elif self.calib_sizes:
            num_low_frequencies, acceleration = self.choose_acceleration()

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_h, num_w, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_h = shape[1]
        num_w = shape[2]
        mask_shape = [1 for _ in shape]
        mask_shape[1] = num_h
        if self.mask_type is MaskType.TWO_D:
            mask_shape[2] = num_w

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_h: int,
        num_w: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_h: Height (number of columns) of k-space (2D subsampling).
            num_w: Width (number of columns) of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """

        num_h = shape[1]
        num_w = shape[2]
        if self.mask_type is MaskType.ONE_D:
            mask = np.zeros(num_h, dtype=np.float32)
            pad = (num_h - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = 1
            assert mask.sum() == num_low_freqs
        elif self.mask_type is MaskType.TWO_D:
            mask = np.zeros([num_h, num_w], dtype=np.float32)
            pad_h = (num_h - num_low_freqs + 1) // 2
            pad_w = (num_w - num_low_freqs + 1) // 2
            mask[
                pad_h : pad_h + num_low_freqs,
                pad_w : pad_w + num_low_freqs
            ] = 1
            assert mask.sum() == num_low_freqs**2
        else:
            raise ValueError('MaskType %s not supported' % self.mask_type)

        return mask

    def choose_acceleration(self):
        """
            Choose acceleration based on class parameters.
            Returns a tuple consisting of either (center_fraction, acceleration)
            or (calib_size, acceleration) depending on which one was specified at
            object creation
        """

        if self.center_fractions:
            list_to_choose_from = self.center_fractions
        elif self.calib_sizes:
            list_to_choose_from = self.calib_sizes

        if self.allow_any_combination:
            return self.rng.choice(list_to_choose_from), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(list_to_choose_from))
            return list_to_choose_from[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other rows are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).
    For 2D case, the same is repeated for the other (column) axis.

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __init__(
        self,
        center_fractions: Sequence[float] = None,
        calib_sizes: Sequence[int] = None,
        accelerations: Sequence[int] = None,
        allow_any_combination: bool = False,
        mask_type: MaskType = None,
    ):
        if mask_type is not MaskType.ONE_D and mask_type is not MaskType.TWO_D:
            raise ValueError(
                'mask_type %s is not supported for RandomMaskFunc.' % mask_type
            )
        super().__init__(
            center_fractions,
            calib_sizes,
            accelerations,
            allow_any_combination,
            mask_type=mask_type,
        )

    def calculate_acceleration_mask(
        self,
        num_h: int,
        num_w: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        if self.mask_type is MaskType.ONE_D:
            prob = (num_h / acceleration - num_low_frequencies) / (
                num_h - num_low_frequencies
            )
            return self.rng.uniform(size=num_h) < prob
        elif self.mask_type is MaskType.TWO_D:
            prob = ((num_h * num_w) / acceleration - num_low_frequencies**2) / (
                num_h * num_w - num_low_frequencies**2
            )
            return self.rng.uniform(size=[num_h, num_w]) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines. Only supports 1D masks.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def __init__(
        self,
        center_fractions: Sequence[float] = None,
        calib_sizes: Sequence[int] = None,
        accelerations: Sequence[int] = None,
        allow_any_combination: bool = False,
        mask_type: MaskType = None,
    ):
        if mask_type is not MaskType.ONE_D:
            raise ValueError(
                'mask_type %s is not supported for EquiSpacedMaskFunc.' % mask_type
            )
        super().__init__(
            center_fractions,
            calib_sizes,
            accelerations,
            allow_any_combination,
            mask_type=mask_type,
        )

    def calculate_acceleration_mask(
        self,
        num_h: int,
        num_w: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_h: Number of 'height dimension' (i.e. rows) of k-space.
            num_w: Number of 'weight dimension' (i.e. columns) of k-space.
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_h, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching. Only supports 1D masks.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __init__(
        self,
        center_fractions: Sequence[float] = None,
        calib_sizes: Sequence[int] = None,
        accelerations: Sequence[int] = None,
        allow_any_combination: bool = False,
        mask_type: MaskType = None,
    ):
        if mask_type is not MaskType.ONE_D:
            raise ValueError(
                'mask_type %s is not supported for EquispacedMaskFractionFunc.' % mask_type
            )
        super().__init__(
            center_fractions,
            calib_sizes,
            accelerations,
            allow_any_combination,
            mask_type=mask_type,
        )

    def calculate_acceleration_mask(
        self,
        num_h: int,
        num_w: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_h: Number of 'height dimension' (i.e. rows) of k-space.
            num_w: Number of 'weight dimension' (i.e. columns) of k-space.
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_h)) / (
            num_low_frequencies * acceleration - num_h
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_h)
        accel_samples = np.arange(offset, num_h - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask

class PoissonDiskMaskFunc(MaskFunc):
    """
    PoissonDiskMaskFunc creates a 2D Poisson disk undersampling mask.
    """

    def __init__(
        self,
        center_fractions: Sequence[float] = None,
        calib_sizes: Sequence[int] = None,
        accelerations: Sequence[int] = None,
        allow_any_combination: bool = False,
        mask_type: MaskType = None,
    ):
        if mask_type is not MaskType.TWO_D:
            raise ValueError(
                'mask_type %s is not supported for PoissonDiskMaskFunc.' % mask_type
            )
        super().__init__(
            center_fractions,
            calib_sizes,
            accelerations,
            allow_any_combination,
            mask_type=mask_type,
        )

    def calculate_acceleration_mask(
        self,
        num_h,
        num_w,
        acceleration,
        offset,
        num_low_frequencies,
    ):

        mask = self.poisson(
            (num_h, num_w),
            acceleration,
            calib=(num_low_frequencies, num_low_frequencies),
            dtype=np.float32,
            tol=0.1,
            max_attempts=20,
        )
        return mask

    # ================================================================ #
    # Adapted from sigpy.
    # Duplicated because nb.jit decorator makes it crash for certain inputs
    # ================================================================ #
    def poisson(self, img_shape, accel, calib=(0, 0), dtype=np.complex64,
                crop_corner=True, return_density=False,
                max_attempts=30, tol=0.1):
        """Generate variable-density Poisson-disc sampling pattern.
        The function generates a variable density Poisson-disc sampling
        mask with density proportional to :math:`1 / (1 + s |r|)`,
        where :math:`r` represents the k-space radius, and :math:`s`
        represents the slope. A binary search is performed on the slope :math:`s`
        such that the resulting acceleration factor is close to the
        prescribed acceleration factor `accel`. The parameter `tol`
        determines how much they can deviate.
        Args:
            img_shape (tuple of ints): length-2 image shape.
            accel (float): Target acceleration factor. Must be greater than 1.
            calib (tuple of ints): length-2 calibration shape.
            dtype (Dtype): data type.
            crop_corner (bool): Toggle whether to crop sampling corners.
            max_attempts (float): maximum number of samples to reject in Poisson
            disc calculation.
            tol (float): Tolerance for how much the resulting acceleration can
                deviate form `accel`.
        Returns:
            array: Poisson-disc sampling mask.
        References:
            Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
            SIGGRAPH sketches. 2007.
        """
        if accel <= 1:
            raise ValueError(f'accel must be greater than 1, got {accel}')

        ny, nx = img_shape
        y, x = np.mgrid[:ny, :nx]
        x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
        x /= x.max()
        y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
        y /= y.max()
        r = np.sqrt(x**2 + y**2)

        slope_max = max(nx, ny)
        slope_min = 0
        while slope_min < slope_max:
            slope = (slope_max + slope_min) / 2
            radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
            radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
            mask = self._poisson(
                img_shape[-1], img_shape[-2], max_attempts,
                radius_x, radius_y, calib)
            if crop_corner:
                mask *= r < 1

            actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

            if abs(actual_accel - accel) < tol:
                break
            if actual_accel < accel:
                slope_min = slope
            else:
                slope_max = slope

        # if abs(actual_accel - accel) >= tol:
        #     raise ValueError(f'Cannot generate mask to satisfy accel={accel}.')

        mask = mask.reshape(img_shape).astype(dtype)

        return mask

    def _poisson(self, nx, ny, max_attempts, radius_x, radius_y, calib):
        mask = np.zeros((ny, nx))

        # initialize active list
        pxs = np.empty(nx * ny, np.int32)
        pys = np.empty(nx * ny, np.int32)
        pxs[0] = self.rng.randint(0, nx)
        pys[0] = self.rng.randint(0, ny)
        num_actives = 1
        while num_actives > 0:
            i = self.rng.randint(0, num_actives)
            px = pxs[i]
            py = pys[i]
            rx = radius_x[py, px]
            ry = radius_y[py, px]

            # Attempt to generate point
            done = False
            k = 0
            while not done and k < max_attempts:
                # Generate point randomly from r and 2 * r
                v = (self.rng.random() * 3 + 1)**0.5
                t = 2 * np.pi * self.rng.random()
                qx = px + v * rx * np.cos(t)
                qy = py + v * ry * np.sin(t)

                # Reject if outside grid or close to other points
                if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                    startx = max(int(qx - rx), 0)
                    endx = min(int(qx + rx + 1), nx)
                    starty = max(int(qy - ry), 0)
                    endy = min(int(qy + ry + 1), ny)

                    done = True
                    for x in range(startx, endx):
                        for y in range(starty, endy):
                            if (mask[y, x] == 1
                                and (((qx - x) / radius_x[y, x])**2 +
                                    ((qy - y) / (radius_y[y, x]))**2 < 1)):
                                done = False
                                break

                k += 1

            # Add point if done else remove from active list
            if done:
                pxs[num_actives] = qx
                pys[num_actives] = qy
                mask[int(qy), int(qx)] = 1
                num_actives += 1
            else:
                pxs[i] = pxs[num_actives - 1]
                pys[i] = pys[num_actives - 1]
                num_actives -= 1

        # Add calibration region
        mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
            int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

        return mask


def create_mask_for_mask_type(
    mask_type_str: str,
    accelerations: Sequence[int],
    center_fractions: Sequence[float] = None,
    calib_sizes: Sequence[int] = None,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        calib_sizes: Size of calibration region.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random1d":
        return RandomMaskFunc(center_fractions, calib_sizes, accelerations, allow_any_combination=True, mask_type=MaskType.ONE_D)
    if mask_type_str == "random2d":
        return RandomMaskFunc(center_fractions, calib_sizes, accelerations, allow_any_combination=True, mask_type=MaskType.TWO_D)
    elif mask_type_str == "equispaced1d":
        return EquiSpacedMaskFunc(center_fractions, calib_sizes, accelerations, allow_any_combination=True, mask_type=MaskType.ONE_D)
    elif mask_type_str == "equispacedfraction1d":
        return EquispacedMaskFractionFunc(center_fractions, calib_sizes, accelerations, allow_any_combination=True, mask_type=MaskType.ONE_D)
    elif mask_type_str == "poisson":
        return PoissonDiskMaskFunc(center_fractions, calib_sizes, accelerations, allow_any_combination=True, mask_type=MaskType.TWO_D)
    else:
        raise ValueError(f"{mask_type_str} not supported")


# ----------------------------------------------------------------------
# Undersampling
# ----------------------------------------------------------------------

def transform_data(input, config_data):

    seed = 0
    if hasattr(config_data, 'seed'):
        seed = config_data.seed

    if hasattr(config_data, 'direction'):
        direction = config_data.direction
    else:
        direction = 'vertical'

    mask_type = config_data.mask_type
    accel = config_data.accel
    if hasattr(config_data, 'mask_path'):
        mask_path = config_data.mask_path

    kspace = input['kspace']
    _, _, nkx, nky = kspace.shape

    if mask_type != 'saved_mask':
        # Calibration region
        center_fraction = None
        calib_size = None
        if hasattr(config_data, 'calib_size'):
            calib_size = [config_data.calib_size]
        elif hasattr(config_data, 'center_fraction'):
            center_fraction = [config_data.center_fraction]
        else:
            if accel == 4:
                center_fraction = [0.08]
            elif accel == 8:
                center_fraction = [0.04]
        
        print('Generating sampling mask')
        mask_func = create_mask_for_mask_type( mask_type_str=mask_type,
                        accelerations=[accel], calib_sizes=calib_size, center_fractions=center_fraction)

        mask, _ = mask_func(shape=[nkx, nky, 1], seed=seed)
        if direction == 'vertical':
            mask = mask[:, None, None].squeeze(-1)
        else:
            mask = mask[None]
            
    else:
        # load mask from a specified path
        mask = torch.from_numpy(np.load(mask_path))
        mask = mask[None,None,:,:]


    kspace *= mask
    input['mask'] = mask

    return

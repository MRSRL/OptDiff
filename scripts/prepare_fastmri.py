"""Prepare the MRI test volumes from the original fastMRI data: the volumes listed in data/lists/fastmri_test.txt (brain
multicoil_test_full, fully sampled), cropped to 384x384, coil-compressed to 8 virtual coils, with ESPIRiT sensitivity
maps and the coil-combined reference image, written to data/fastmri/, where configs/mri/fastmri.yaml points.

    TOOLBOX_PATH=/path/to/bart python scripts/prepare_fastmri.py /path/to/multicoil_test_full

The sensitivity maps need BART (https://mrirecon.github.io/bart/): TOOLBOX_PATH is its folder, with the bart binary
and the Python wrapper in python/. Run from the repository root; volumes already written are skipped.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from utils.degradations_mri import SenseOperator, fftn, ifftn

# BART: TOOLBOX_PATH is its folder, with the bart binary and the Python wrapper in python/
if 'TOOLBOX_PATH' not in os.environ:
    sys.exit('Set TOOLBOX_PATH to the BART folder (https://mrirecon.github.io/bart/)')
sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
# BART's OpenBLAS can crash on machines with more cores than its build allows; cap its threads unless set
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import bart

IMAGE_SHAPE = (384, 384)
NUM_VIRTUAL_COILS = 8
NUM_CALIB = 128  # calibration region of ESPIRiT


def read_list():
    with open(os.path.join('data', 'lists', 'fastmri_test.txt')) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def locate(source, names):
    """Path of every volume, in source itself or anywhere below it."""
    paths = {name: os.path.join(source, name) for name in names if os.path.isfile(os.path.join(source, name))}
    missing = set(names) - set(paths)
    if missing:
        for root, _, files in os.walk(source):
            paths.update({f: os.path.join(root, f) for f in missing.intersection(files)})
    missing = sorted(set(names) - set(paths))
    if missing:
        sys.exit(f'{len(missing)} of the {len(names)} volumes are not in {source}, e.g. {missing[0]}')
    return paths


def preprocess_slice(kspace, im_shape):
    """Pre-process k-space data.

    Args:
        kspace: Shape `#coils x H x W`
        im_shape (tuple): Shape to crop to.

    Returns:
        ndarray: Shape `#coils x H x W`
    """
    # Pre-process k-space data (PyTorch)
    #   1. Reconstruct fully sampled image
    #   2. Crop to (yres, xres)
    #   3. Take fourier transform
    # Note this preprocessing technique does not work for
    # test/challenge images as scans in these datasets are undersampled.
    kspace_tensor = torch.from_numpy(kspace)  # (1, 640, 372, 15, 2)
    image_tensor = ifftn(kspace_tensor, dim=[-1,-2])
    image_tensor = complex_center_crop_2d(image_tensor, im_shape)
    kspace_tensor = fftn(image_tensor, dim=[-1,-2])
    kspace_slice = kspace_tensor.clone().numpy()  # (640, 372, 15)

    return kspace_slice


def complex_center_crop_2d(data, shape):
    """
    Util function from meddlr

    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


# Geometric coil compression (gcc) [1] and ESPIRiT sensitivity maps [2] with BART, from the dl-cs-dynamic repository [3]
# [1] Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for accelerated imaging with Cartesian sampling.
#     Magn Reson Med 2013 Mar 9;69:571-582
# [2] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT - an eigenvalue approach
#     to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med 2014. Mar;71(3):990-1001.
# [3] Sandino CM, Lai P, Vasanawala SS, Cheng JY. Accelerating cardiac cine MRI using a deep learning-based ESPIRiT
#     reconstruction. Magn Reson Med 2021. Jan;85(1):152-67.

def calc_gcc_weights_c(ks_calib, num_virtual_channels, correction=True):
    """Calculate coil compression weights.
    Input
      ks_calib -- raw k-space data of dimensions
                    (num_channels, num_readout, num_kx)
      num_virtual_channels -- number of virtual channels to compress to
      correction -- apply rotation correction (default: True)
    Output
      cc_mat -- coil compression matrix (use apply_gcc_weights)
    """
    me = "coilcomp.calc_gcc_weights_c"

    num_kx = ks_calib.shape[2]
    # num_readout = ks_calib.shape[1]
    num_channels = ks_calib.shape[0]

    if num_virtual_channels > num_channels:
        print(["%s> Num of virtual channels (%d) is more than the actual "
               + " channels (%d)!"] % (me, num_virtual_channels, num_channels))
        return np.eye(num_channels, dtype=np.complex64)

    ks_calib_int = ks_calib.copy()

    cc_mat = np.zeros((num_virtual_channels, num_channels, num_kx),
                      dtype=np.complex64)
    for i_x in range(num_kx):
        ks_calib_x = np.squeeze(ks_calib_int[:, :, i_x])
        U, s, Vh = np.linalg.svd(ks_calib_x.T, full_matrices=False)
        V = Vh.conj()
        cc_mat[:, :, i_x] = V[0:num_virtual_channels, :]

    if correction and num_kx > 1:
        for i_x in range(int(num_kx/2)-2, -1, -1):
            V1 = cc_mat[:, :, i_x+1]
            V2 = cc_mat[:, :, i_x]
            A = np.matmul(V1.conj(), V2.T)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj()
            cc_mat[:, :, i_x] = np.matmul(P, cc_mat[:, :, i_x])

        for i_x in range(int(num_kx/2)-1, num_kx, 1):
            V1 = cc_mat[:, :, i_x-1]
            V2 = cc_mat[:, :, i_x]
            A = np.matmul(V1.conj(), V2.T)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj()
            cc_mat[:, :, i_x] = np.matmul(P, np.squeeze(cc_mat[:, :, i_x]))

    return cc_mat

def apply_gcc_weights_c(ks, cc_mat):
    """Apply coil compression weights.
    Input
      ks -- raw k-space data of dimensions (num_channels, num_readout, num_kx)
      cc_mat -- coil compression matrix calculated using calc_gcc_weights
    Output
      ks_out -- coil compresssed data
    """
    me = "coilcomp.apply_gcc_weights_c"

    num_channels = ks.shape[0]
    num_readout = ks.shape[1]
    num_kx = ks.shape[2]
    num_virtual_channels = cc_mat.shape[0]

    if num_channels != cc_mat.shape[1]:
        print("%s> ERROR! num channels does not match!" % me)
        print("%s>   ks: num channels = %d" % (me, num_channels))
        print("%s>   cc_mat: num channels = %d" % (me, cc_mat.shape[1]))

    ks_out = np.zeros((num_virtual_channels, num_readout, num_kx),
                      dtype=np.complex64)
    for i_channel in range(num_virtual_channels):
        cc_mat_i = np.reshape(cc_mat[i_channel, :, :],
                              (num_channels, 1, num_kx))
        ks_out[i_channel, :, :] = np.sum(ks * cc_mat_i, axis=0)

    return ks_out

def coil_compress(kspace, num_virtual_coils=6, cc_mat=None):
    """
    Performs coil compression slice-by-slice on 2D cine data
    Input 
        kspace: (num_slices, num_coils, num_kx, num_ky)
        num_virtual_coils: number of virtual coils after compression
    Output
        kspace: (num_slices, num_virtual_coils, num_kx, num_ky)
        cc_mat: coil compression matrix
    """
    # Get data dimensions
    num_slices, num_coils, num_kx, num_ky = kspace.shape

    if num_virtual_coils < num_coils:
        d = np.transpose(kspace, [1, 2, 3, 0]) #coils x H x W x #slices
        d = np.reshape(d, [num_coils, num_kx*num_ky, num_slices])
        # cc code expects (num_channels, num_readout, num_kx)
        if cc_mat is None:
            cc_mat = calc_gcc_weights_c(d, num_virtual_coils)
    
        dsl = apply_gcc_weights_c(d, cc_mat)
        # above returns (num_vchannels, num_readout, num_kx)
        kspace = np.reshape(dsl, [num_virtual_coils, num_kx, num_ky, num_slices])
        kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace, cc_mat


def ecalib(kspace, num_calib=20, crop_value=0):
    """
    Calls BART's ESPIRiT under the hood to estimate sensitivity maps slice-by-slice.
    """

    num_slices, num_coils, num_kx, num_ky = kspace.shape
    maps = np.zeros((num_slices, num_coils, num_kx, num_ky), dtype=np.complex64)

    for sl in range(num_slices):
        # Perform ESPIRiT calibration in BART
        # maps_sl = bart.bart(1, f'ecalib -S -m 1 -r {num_calib} -c {crop_value}', \
        maps_sl = bart.bart(1, f'ecalib -a -m 1 -r {num_calib}', \
                    kspace[sl, ..., None].transpose([1, 2, 3, 0])) # H x W x 1 x #coils
        maps[sl, ...] = np.squeeze(maps_sl).transpose([2, 0, 1])

    return maps


def prepare(path, out):
    with h5py.File(path, 'r') as f:
        kspace = f['kspace'][()]
        shape = f['reconstruction_rss'].shape[1:3]  # matrix size
    if tuple(shape) != IMAGE_SHAPE:
        sys.exit(f'{path}: matrix size {shape}, expected {IMAGE_SHAPE}')
    kspace = preprocess_slice(kspace, shape)  # crop the coil images to the matrix size
    kspace_cc, _ = coil_compress(kspace, num_virtual_coils=NUM_VIRTUAL_COILS)
    s_maps = ecalib(kspace_cc, num_calib=NUM_CALIB)
    ref = SenseOperator(torch.from_numpy(s_maps))(torch.from_numpy(kspace_cc), adjoint=True).numpy()
    with h5py.File(out + '.tmp', 'w') as f:
        f.create_dataset('s_maps', data=s_maps)
        f.create_dataset('kspace', data=kspace_cc)
        f.create_dataset('ref', data=ref)
    os.replace(out + '.tmp', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='folder with the fastMRI brain multicoil_test_full volumes')
    args = parser.parse_args()

    if not os.path.exists('main_mri.py'):
        sys.exit('Run from the repository root.')

    names = read_list()
    paths = locate(args.source, names)
    os.makedirs(os.path.join('data', 'fastmri'), exist_ok=True)
    for i, name in enumerate(names):
        out = os.path.join('data', 'fastmri', name)
        if os.path.exists(out):
            continue
        start = time.time()
        prepare(paths[name], out)
        print(f'[{i + 1}/{len(names)}] {out} ({time.time() - start:.0f} s)', flush=True)


if __name__ == '__main__':
    main()

"""MRI data: loading the prepared slices, undersampling and scaling them, and the operators of a run."""
import numpy as np
import torch
import h5py
import glob
import os
from .degradations_mri import SenseOperator, transform_data

to_cplx = lambda x: torch.complex(x[:, 0], x[:, 1])
to_real = lambda x: torch.view_as_real(x).permute(0,3,1,2) # Batch x channels x H x W

def load_data(data_config):
    input_dir = data_config.input_dir
    file_idx = data_config.file
    slice_idx = data_config.slice

    file_list = sorted(glob.glob(input_dir + '/*.h5'))

    print('List of files in {:s}'.format(input_dir))
    for i in range(min(10,len(file_list))):
        file = os.path.basename(file_list[i])
        print('[{:d}] {:s}'.format(i, file))

    print(' ')
    file = file_list[file_idx]
    file_base = os.path.basename(file)
    print('Reading file [{:d}] {:s}'.format(file_idx, file_base))

    with h5py.File(file, "r") as f:
        kspace = f["kspace"][()]
        ref = f["ref"][()]
        s_maps = f["s_maps"][()]

    # Check if slice_idx is int or list
    if isinstance(slice_idx, int):
        slice_idx = [slice_idx]

    kspace = torch.from_numpy(kspace[slice_idx,...]).type(torch.complex64)
    ref = torch.from_numpy(ref[slice_idx,...]).type(torch.complex64)
    s_maps = torch.from_numpy(s_maps[slice_idx,...]).type(torch.complex64)

    input_data = {'kspace': kspace, 's_maps': s_maps, 'ref': ref}
    return input_data


# ----------------------------------------------------------------------
# Undersampling, scaling and SENSE operators
# ----------------------------------------------------------------------

def normalize_image(image):
    size_img = image.shape
    abs_image = torch.flatten(image.abs(), start_dim=1)
    scaling = torch.quantile(abs_image, 0.99, dim=1)
    scaling = scaling.reshape([size_img[0]] + (len(size_img)-1)*[1])
    norm_image = image / scaling
    return norm_image, scaling

def mri_data_prep(input, device, config_data):
    
    transform_data(input, config_data) # undersampling
    kspace = input['kspace'].to(device)
    ref = input['ref'].to(device)
    maps = input['s_maps'].to(device)
    mask = input['mask'].to(device)

    # Scaling kspace
    ref, scaling = normalize_image(ref)
    kspace /= scaling[:, None]

    # Models
    mri_model = SenseOperator(maps, mask)
    forward_operator = lambda x: mri_model(to_cplx(x))
    adjoint_operator = lambda x: to_real(mri_model(x, adjoint=True))

    # Add noise 
    if hasattr(config_data, 'add_noise') and config_data.add_noise: 
        noise = torch.randn_like(kspace) * config_data.noise_level 
        kspace += noise * mask

        # new reference 
        mri_model_fs = SenseOperator(maps)
        ref += mri_model_fs(noise, adjoint=True)

        # Scale again 
        ref, scaling = normalize_image(ref)
        kspace /= scaling[:, None]

    # Dictionary
    data = {'y': kspace, 'ref': ref, 'mask': mask}
    operators = {'forward': forward_operator, 'adjoint': adjoint_operator, 'mri_model': mri_model}

    return data, operators

def save_mask(mask, filename):
    mask = mask.cpu().numpy()
    np.savez(filename, mask)
    print(f"Mask saved to {filename}")
    return mask

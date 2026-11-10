"""DDPM noise schedule, sigma/alpha conversion and image pre/postprocessing."""
import numpy as np
import torch


class Diffusion:
    def __init__(self, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=1000, given_betas=None):
        # from utils.functions import sigmoid
        if given_betas is None:
            if beta_schedule == "quad":
                betas = (
                    np.linspace(
                        beta_start**0.5,
                        beta_end**0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif beta_schedule == "linear":
                betas = np.linspace(
                    beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "const":
                betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
                )
            # elif beta_schedule == "sigmoid":
            #     betas = np.linspace(-6, 6, num_diffusion_timesteps)
            #     betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
            else:
                raise NotImplementedError(beta_schedule)
            assert betas.shape == (num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = given_betas
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
    def alpha(self, t):
        return self.alphas.index_select(0, t+1)


def postprocess(x):
    if type(x) == list:
        return [(v + 1) / 2 for v in x]
    else:
        return (x + 1) / 2


def preprocess(x):
    return x * 2 - 1


def sig2alpha(sigmas):
    return 1 / (sigmas**2 +1)

def alpha2sig(alphas):
    a= (1-alphas)/alphas
    return a.sqrt()

def getclosest(long_arr, short_arr):
    closest_ind = ((long_arr.unsqueeze(0) - short_arr.unsqueeze(1))**2).argmin(dim=1)
    closest_values = long_arr[closest_ind]
    return closest_values, closest_ind


def uniform_timesteps(num_steps, start_step, end_step, num_diffusion_timesteps):
    """The standard schedule (noise_schedule_config type uniform): num_steps evenly spaced network timesteps in
    ascending order, as in DDIM and RED-diff: end_step, end_step + skip, ..., with skip = (start_step - end_step) //
    num_steps."""
    if not 0 <= end_step < start_step <= num_diffusion_timesteps:
        raise ValueError(f'The uniform schedule needs 0 <= end_step < start_step <= {num_diffusion_timesteps} '
                         f'(the network\'s timesteps), not end_step {end_step}, start_step {start_step}.')
    skip = (start_step - end_step) // num_steps
    timesteps = list(range(end_step, start_step, skip)) if skip > 0 else []
    if len(timesteps) != num_steps:
        raise ValueError(f'The uniform schedule from start_step {start_step} to end_step {end_step} gives '
                         f'{len(timesteps)} timesteps, not num_steps {num_steps}: start_step - end_step needs to be a '
                         f'multiple of num_steps.')
    return timesteps


def edm_timesteps(alphas, num_steps, sigma_max, sigma_min, rho):
    """The EDM schedule (noise_schedule_config type edm): num_steps noise levels from sigma_max down to sigma_min, each
    snapped to the closest noise level of the network (repeats kept). alphas are Diffusion.alphas, where index t + 1
    holds alpha_bar of network timestep t. Returns, per step from the highest noise level down, alpha_bar of the
    snapped level and its network timestep."""
    step_inds = torch.arange(num_steps, dtype=torch.float32, device=alphas.device)
    sigmas = (sigma_max ** (1 / rho) + step_inds / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    closest_values, closest_ind = getclosest(alphas, sig2alpha(sigmas))
    if (closest_ind == 0).any():
        raise ValueError(f'sigma_min {sigma_min} is below the lowest noise level of the network '
                         f'({alpha2sig(alphas[1]).item():.4f}): it snaps to the clean image, which is no network timestep.')
    return closest_values, closest_ind - 1

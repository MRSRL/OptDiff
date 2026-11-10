import hydra
import os
import dnnlib
import pickle
import torch
import tqdm
from omegaconf import open_dict
from utils.data_mri import load_data, mri_data_prep, to_cplx, to_real
from algo.optimizers_mri import FirstOrder
from algo.baselines_mri import DPS, PiGDM, DAPS, DiffPIR
from utils.output_mri import Outputs, Debugger
from utils.timing import clock, save_runtime, runtime_summary
from utils.task_config import missing_keys


@hydra.main(config_path='configs/mri', version_base=None)
def main(config):
    # Values marked ??? are only checked when read: list them all before loading the data and the network
    missing = missing_keys(config)
    if missing:
        raise ValueError('Values need to be provided for: ' + ', '.join(missing))

    # Configs without an algo field run OptDiff
    with open_dict(config.optimizer_config):
        if 'algo' not in config.optimizer_config:
            config.optimizer_config.algo = 'OptDiff'

    if config.optimizer_config.algo == 'OptDiff' and config.optimizer_config.weights == 'heuristics' and \
            ('optimizer_weights_config' not in config or config.optimizer_weights_config is None):
        raise ValueError("Heuristic weights need optimizer_weights_config: a file in configs/mri/optimizer_weights_config "
                         "(optimizer_weights_config=<name>, e.g. example_weights), or the values, e.g. 'optimizer_weights_config="
                         "{parameterization:normalized,alpha:{function:exp,parameters:[a,b,c]},"
                         "lamda:{function:constant,parameters:d},beta:{function:constant,parameters:e}}' (beta with momentum)")
    print(config)

    """ setup """
    input_data = load_data(config.data_config)
    slices = config.data_config.slice
    if isinstance(slices, int):
        slices = [slices]
    nslices = input_data['kspace'].shape[0]

    """ network """
    # Load network.
    network_pkl = os.path.join(config.model_config.directory,config.model_config.name)
    device = torch.device(config.model_config.device)
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    net.eval()
    
    """ noise schedule """
    config_noise = config.noise_schedule_config
    num_steps = config_noise.num_steps
    sigma_min = config_noise.sigma_min
    sigma_max = config_noise.sigma_max
    rho = config_noise.rho
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    noise_schedule = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if config.optimizer_config.algo == 'daps':
        noise_schedule = torch.cat([noise_schedule, torch.zeros(1, device=device)])  # DAPS needs one additional 0 step
    
    if hasattr(config_noise, 'repeat'):
        # repeat each entry a few times to increase the number of steps
        noise_schedule = torch.repeat_interleave(noise_schedule, config_noise.repeat)
        num_steps = len(noise_schedule)
        config_noise.num_steps = num_steps
        # setattr(config_noise, "noise_schedule", noise_schedule)

    # Optimizer/sampler selection
    optimizer_algo = config.optimizer_config.algo
    optimizer_class = None
    if optimizer_algo == 'OptDiff':
        order = config.optimizer_config.order
        if order == 1:
            optimizer_class = FirstOrder
        else:
            raise ValueError(f'Unknown order {order}. Only first-order OptDiff is supported.')
    elif optimizer_algo == 'dps':
        optimizer_class = DPS
    elif optimizer_algo == 'pigdm':
        optimizer_class = PiGDM
    elif optimizer_algo == 'daps':
        optimizer_class = DAPS
    elif optimizer_algo == 'diffpir':
        optimizer_class = DiffPIR
    else:
        raise ValueError(f'Unknown optimizer/sampler {optimizer_algo}.')

    runtimes = []
    for isl in range(nslices):
        print(f'Processing slice {isl+1}/{nslices}...')

        # Get slice of each key in input_data dictionary using list comprehension
        sl_input_data = {key: value[[isl],...] for key, value in input_data.items()}
        config.data_config.slice = slices[isl]

        # Prepare data
        data, operators = mri_data_prep(sl_input_data, device, config.data_config)

        # Optimizer
        with open_dict(config.optimizer_config):
            config.optimizer_config.num_steps = num_steps
        weights_config = None
        if hasattr(config, 'optimizer_weights_config'): #Checking if there config for weights heuristics
            weights_config = config.optimizer_weights_config
        ref = None
        # Oracle weights need the reference image
        if config.optimizer_config.weights in ('optimal', 'optimal_cnst_lamda') or getattr(config.optimizer_config, 'reddiff_do_optimal_step', False):
            ref = data['ref']

        if optimizer_algo == 'daps':
            optimizer = optimizer_class(data['y'], operators, net, config.optimizer_config, weights_config=weights_config, ref=ref, noise_schedule_config=config_noise)
        else:
            optimizer = optimizer_class(data['y'], operators, net, config.optimizer_config, weights_config=weights_config, ref=ref)

        # Output and Debug objects
        OutputObj = Outputs(config)
        if config.debug_config.debug:
            debugger = Debugger(config, OutputObj.output_dir, optimizer, ref=data['ref'], add_to_fname=OutputObj.add_to_fname)

        # Reconstruction loop (timed, see utils/timing.py)
        start_time = clock(device)
        for i in tqdm.tqdm(range(num_steps), unit='step'): # 0, ..., N-1
            noise_std = noise_schedule[i]
            # Provide next sigma for if available
            if (i + 1) < num_steps and hasattr(optimizer, 'set_next_sigma'):
                optimizer.set_next_sigma(noise_schedule[i+1])
            elif (i + 1) == num_steps and optimizer_algo == 'daps':
                optimizer.set_next_sigma(noise_schedule[i+1])
            optimizer.estimate_update(noise_std)
            # optimizer.estimate_weights()
            optimizer.step()

            if config.debug_config.debug:
                debugger.save_debug_outputs(i)
        runtimes.append(clock(device) - start_time)


        # Save outputs
        output_data = optimizer.prepare_output(to_real(data['ref']))
        OutputObj.load_output_data(output_data, mask=data['mask'])
        psnr_rec, psnr_abs_rec, ssim_rec = OutputObj.estimate_metrics()
        OutputObj.save_outputs(timing=runtimes[-1])
        save_runtime(OutputObj.output_dir, runtimes[-1])

        print(f'\tPSNR: {psnr_rec:.2f} dB, PSNR abs: {psnr_abs_rec:.2f} dB, SSIM: {ssim_rec:.4f}, time: {runtimes[-1]:.4f} s')

    print(runtime_summary(runtimes))


if __name__ == '__main__':
    main()

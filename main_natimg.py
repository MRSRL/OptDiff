import hydra
import os
import torch
import tqdm
from algo.optimizers_natimg import FirstOrder
from algo.baselines_natimg import DiffusionModel, build_baseline
from utils.output_natimg import Outputs, Debugger, prepare_output_data, config_to_log
from utils.task_config import missing_keys
from hydra.utils import call
from utils.timing import clock, save_runtime, runtime_summary
from utils.diffusion_natimg import *
from utils.degradations_natimg import build_degradation_model
import numpy as np

@hydra.main(config_path='configs/natimg', version_base=None)
def main(config):
    # OptDiff (optimizer_config) or a baseline (baseline_config), selected with algo=<name>, see configs/natimg/algo.
    # Configs without an algo field run OptDiff.
    algo = config.algo if 'algo' in config else 'OptDiff'

    # Noise schedule (configs/natimg/noise_schedule_config): uniform (the baselines' standard schedule) or edm
    schedule_type = config.noise_schedule_config.get('type')
    if schedule_type not in ('uniform', 'edm'):
        problem = 'is missing' if schedule_type is None else f'is {schedule_type}'
        raise ValueError(f'noise_schedule_config.type {problem}; it must be uniform or edm: select a file in '
                         'configs/natimg/noise_schedule_config (noise_schedule_config=standard, edm or example_schedule), '
                         'or give the type in your own file')
    if algo == 'OptDiff' and schedule_type != 'edm':
        raise ValueError('OptDiff needs an EDM noise schedule: noise_schedule_config=edm (its default) with '
                         'noise_schedule_config.sigma_max, sigma_min and rho, or a file such as noise_schedule_config=example_schedule')

    # Values marked ??? are only checked when read: list them all before loading the data and the network
    missing = missing_keys(config)
    if missing:
        raise ValueError('Values need to be provided for: ' + ', '.join(missing))

    if algo == 'OptDiff' and config.optimizer_config.weights == 'heuristics' and \
            ('optimizer_weights_config' not in config or config.optimizer_weights_config is None):
        raise ValueError("Heuristic weights need optimizer_weights_config: a file in configs/natimg/optimizer_weights_config "
                         "(optimizer_weights_config=<name>, e.g. example_weights), or the values, e.g. 'optimizer_weights_config="
                         "{alpha:{function:linear,parameters:[a,b]},lamda:{function:constant,parameters:c},"
                         "beta:{function:constant,parameters:d}}' (beta with momentum)")
    print(config_to_log(config))

    """ setup """
    dataset = call(config.data_config.loader)  # the dataset config names its loader

    print("Dataset size: ", len(dataset))
    slices = config.data_config.slice

    deg = config.data_config.deg

    # seed = config.optimizer_config.seed

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)



    """ network """
    # Load network.
    device = torch.device(config.model_config.device)
    # print(device)

    net = call(config.model)
    print(f"Loading model from {config.model_config.checkpoint}..")
    net.load_state_dict(torch.load(config.model_config.checkpoint, map_location=device)) ## loaded succesfully
    net.to(device)
    net.eval()
    # print(net)

    #### degradation
    H = build_degradation_model(deg=deg, image_size=256, channels=3, device=device)


    """ noise schedule """
    config_noise = config.noise_schedule_config
    num_steps = config_noise.num_steps
    print(config_noise)
    # Noise levels of the network: its DDPM schedule (model_config)
    config_model = config.model_config
    diffusion = Diffusion(config_model.beta_schedule, config_model.beta_start, config_model.beta_end,
                          config_model.num_diffusion_timesteps)
    if schedule_type == 'edm':
        # Per step, from the highest noise level down: alpha_bar of the snapped noise level and its network timestep
        alpha_steps, t_steps = edm_timesteps(diffusion.alphas.to(device), num_steps, config_noise.sigma_max,
                                             config_noise.sigma_min, config_noise.rho)

    if algo == 'OptDiff':
        # Optimizer class
        order = config.optimizer_config.order
        if order != 1:
            raise ValueError(f'Unknown order {order}.')
        optimizer_class = FirstOrder

        weights_config = None
        if hasattr(config, 'optimizer_weights_config'): #Checking if there config for weights heuristics
            weights_config = config.optimizer_weights_config
        print(weights_config)
    else:
        # The baselines run their own sampling loop on the network timesteps of the schedule, in ascending order
        if schedule_type == 'edm':
            step_indices = t_steps.flip(0).tolist()
        else:
            step_indices = uniform_timesteps(num_steps, config_noise.start_step, config_noise.end_step,
                                             config_model.num_diffusion_timesteps)
        print(step_indices)

        if config.debug_config.debug:
            raise ValueError('The debugger only supports OptDiff.')
        # Diffusion keeps its tables on the default GPU; the samplers index them on the network's device
        diffusion.alphas = diffusion.alphas.to(device)
        diffusion.betas = diffusion.betas.to(device)
        baseline = build_baseline(algo, DiffusionModel(net, diffusion), H, config.data_config.input_noise, config.baseline_config)

    if slices == 'all':
        slices = range(len(dataset))
    elif type(slices) == int:
        slices = range(slices)

    psnrs = []
    ssims = []
    lpips = []
    times = []
    for img in slices:
        print(f"Running on img no {img} ...")
        # start_it = time.time()
        input_data = dataset[img] # 0-6
        config.data_config.slice = img
        #### just returning images here
        print(input_data.shape)

        # Prepare data
        # data, operators = mri_data_prep(sl_input_data, device, config.data_config)
        # data = {'y': kspace, 'ref': ref, 'mask': mask}
        # operators = {'forward': forward_operator, 'adjoint': adjoint_operator, 'mri_model': mri_model}

        sigma_y = config.data_config.input_noise
        x = input_data[None]
        if deg.startswith("inpainting_"):
            H.set_indices(torch.tensor(0)) # we only have one mask but this is for mask selection if there is one for each img
        elif "rbox" in deg:
            H.set_indices(torch.tensor(img)) # deterministic per-image box location, keyed by dataset index
        x = preprocess(x).to(device)

        # Seed per image, before the measurement: every image gets its own measurement and sampling noise,
        # and its result does not depend on which other images are run
        seed = config.optimizer_config.seed if algo == 'OptDiff' else config.baseline_config.seed
        torch.manual_seed(seed + img)
        y_0 = H.H(x)
        y_0 = y_0 + torch.randn_like(y_0) * sigma_y
        data = {'y': y_0, 'ref': x}

        # Output object
        OutputObj = Outputs(config)
        os.makedirs(OutputObj.output_dir, exist_ok=True)

        # Reconstruction (timed): recon, and init for methods that start from an estimate, are in [-1, 1]
        init = None
        if algo == 'OptDiff':
            ref = data['ref'] #None
            if config.optimizer_config.weights == 'optimal':  # For optimal weights we need to reference image
                ref = data['ref']
            # y, H, denoiser, config, weights_config=None, ref=None
            optimizer = optimizer_class(data['y'], H, net, config.optimizer_config, weights_config=weights_config, ref=ref,
                                        reseed=False)  # continue the image's random stream seeded above
            if config.debug_config.debug:
                debugger = Debugger(config, OutputObj.output_dir, optimizer, ref=data['ref'], add_to_fname=OutputObj.add_to_fname)

            lamda = config.optimizer_config.lamda ## constant lamda hat, others optimal
            # print(f"Using lamda: {lamda}")
            # Reconstruction loop (timed, see utils/timing.py)
            start_it = clock(device)
            for i in tqdm.tqdm(range(num_steps), unit='step'): # 0, ..., N-1
                noise_std = alpha_steps[i]  # alpha_bar of the step's noise level
                t = t_steps[i]  # its network timestep

                if lamda is None or config.optimizer_config.weights == 'heuristics':
                    optimizer.estimate_update(noise_std, t)
                else:
                    # print("here")
                    optimizer.estimate_update_cnst_lamda(noise_std, t, lamda)
                # optimizer.estimate_weights()
                optimizer.step()

                if config.debug_config.debug:
                    debugger.save_debug_outputs(i)
            times.append(clock(device) - start_it)
            recon, init = optimizer.x, optimizer.x0
        else:
            # The sampler runs its own loop (x only gives the shape and device)
            start_it = clock(device)
            xt_s, _ = baseline.sample(x, None, step_indices, y_0=data['y'])
            times.append(clock(device) - start_it)
            recon = xt_s[0].to(device)

        # Save outputs
        output_data = prepare_output_data(recon, data['ref'], data['y'], init=init)
        OutputObj.load_output_data(output_data, mask=None)
        psnr_rec, ssim_rec, lpips_rec = OutputObj.estimate_metrics()
        OutputObj.save_outputs(iter=img, timing=times[-1])
        save_runtime(OutputObj.output_dir, times[-1])
        psnrs.append(psnr_rec)
        ssims.append(ssim_rec)
        lpips.append(lpips_rec)

        print(f"PSNR rec: {psnr_rec:.2f} dB, SSIM: {ssim_rec:.4f}, LPIPS: {lpips_rec:.4f}, time: {times[-1]:.4f} s")

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    lpips = np.array(lpips)

    psnr_mean = np.nanmean(psnrs)
    ssim_mean = np.nanmean(ssims)
    lpips_mean = np.nanmean(lpips)
    psnr_std = np.nanstd(psnrs)
    ssim_std = np.nanstd(ssims)
    lpips_std = np.nanstd(lpips)

    print(f"Average PSNR: {psnr_mean:.4f} +- {psnr_std:.4f}, SSIM: {ssim_mean:.4f} +- {ssim_std:.4f}, LPIPS: {lpips_mean:.4f} +- {lpips_std:.4f}")
    print(runtime_summary(times))

    OutputObj.add_end_message([ "------Overall metrics------",
        f"Average PSNR: {psnr_mean:.4f} +- {psnr_std:.4f}",
        f"Average SSIM: {ssim_mean:.4f} +- {ssim_std:.4f}",
        f"Average LPIPS: {lpips_mean:.4f} +- {lpips_std:.4f}",
        runtime_summary(times)])
    # OptDiff: indices into Diffusion.alphas (network timestep + 1); baselines: network timesteps
    OutputObj.add_end_message(["sampled indices:",
                                f"{t_steps + 1 if algo == 'OptDiff' else step_indices}"])

if __name__ == '__main__':
    main()

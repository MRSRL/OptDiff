import PIL.Image
import h5py
import torch
from utils.metrics import LPIPSMetric, psnr, ssim, nrmse, psnr_abs
from .degradations_mri import fftn
import os, sys
import datetime
import numpy as np
from omegaconf import OmegaConf
from utils.diffusion_natimg import *

class Debugger:
    def __init__(self, config, debug_dir, ReconApp, ref=None, add_to_fname=''):
        self.config = config
        self.debug_dir = debug_dir
        self.ReconApp = ReconApp
        self.nbatch, _, self.nx, self.ny = self.ReconApp.x.shape
        self.device = self.ReconApp.device
        self.denoiser = self.ReconApp.denoiser
        self.ref = check_cplx(ref)
        self.add_to_fname = add_to_fname

        self.config_debug = config.debug_config
        self.config_data = config.data_config
        self.config_noise = config.noise_schedule_config
        self.config_output = config.output_config
        self.config_optimizer = config.optimizer_config

        self.debug_noise_levels = self.config_debug.num_noise_levels
        self.debug_outputs = None
        self.debug_filename = os.path.join(self.debug_dir, self.config_debug.debug_filename + self.add_to_fname)
        self._get_noise_schedule()    

        if self.config_debug.save_loss_multifreq:
            self.num_freq_bands = self.config_debug.num_freq_levels
            self._init_multifreq_loss()

        self._init_debug_outputs()    
        return
    
    def _init_debug_outputs(self):
        num_steps = self.config.noise_schedule_config.num_steps
        nweights = 2 
        if self.config_optimizer.momentum:
            nweights += 1
        if self.config_optimizer.order == 2:
            nweights += 1

        self.losses = {'loss_obs': np.zeros((num_steps + 1, self.nbatch), dtype=np.float32), 
                       'loss_denoise': np.zeros((num_steps + 1, self.nbatch), dtype=np.float32)}
        if self.config_debug.save_loss_denoise_multilevel:
            self.losses['loss_denoise_multilevel'] = np.zeros((num_steps + 1, self.nbatch, self.debug_noise_levels), dtype=np.float32)
        if self.config_debug.save_loss_multifreq:
            self.losses['loss_multifreq'] = np.zeros((num_steps + 1, self.nbatch, self.num_freq_bands), dtype=np.float32)

        self.metrics = {'psnr': np.zeros((num_steps + 1, self.nbatch), dtype=np.float32),
                        #'psnr_abs': np.zeros((num_steps + 1, self.nbatch), dtype=np.float32),
                        'ssim': np.zeros((num_steps + 1, self.nbatch), dtype=np.float32)}
        
        if self.config_debug.save_gradients:
            self.gradients = {'grad_obs': np.zeros((num_steps + 1, *self.ReconApp.x.shape), dtype=np.float32),
                              'grad_denoise': np.zeros((num_steps + 1, *self.ReconApp.x.shape), dtype=np.float32)}
        
        self.nnls_flag = {'nnls_flag': np.zeros((num_steps + 1), dtype=bool)}
        self.weights = {'weights': np.zeros((num_steps + 1, nweights), dtype=np.float32)}

        self.save_debug_outputs(-1)

        return
    
    def _init_multifreq_loss(self):
        # Grid with the same size as the image with polar coordinates
        h, w = self.nx, self.ny
        y = torch.arange(h, device=self.device).view(-1, 1).expand(h, w)
        x = torch.arange(w, device=self.device).view(1, -1).expand(h, w)
        y = y - h // 2
        x = x - w // 2
        r = torch.sqrt(x**2 + y**2)

        # Divice into frequency bands
        bands = torch.linspace(0, self.nx//2, self.num_freq_bands + 1, device=r.device)
        masks = torch.zeros((self.num_freq_bands, *r.shape), device=r.device)
        for i in range(self.num_freq_bands):
            masks[i] = ((r >= bands[i]) & (r < bands[i + 1])).float()
            
        self.multifreq_mask = masks[None, ...]
        self.fftn = lambda x: fftn(x, dim=(-2, -1), norm='ortho')


        self.ref_ksp = self.fftn(check_cplx(self.ref))[:, None] * self.multifreq_mask
        self.ref_ksp = self.ref_ksp.view(self.num_freq_bands * self.nbatch, self.nx, self.ny)
        return

    def _get_multifreq_loss(self):
        x_ksp = self.fftn(self.x_cplx)[:, None] * self.multifreq_mask
        x_ksp = x_ksp.view(self.nbatch * self.num_freq_bands, self.nx, self.ny)

        print(self.ref_ksp.shape, x_ksp.shape)
        loss_multifreq = nrmse(self.ref_ksp, x_ksp).view(self.nbatch, self.num_freq_bands)
        
        return loss_multifreq
    
    def _get_metrics(self):
        ref = postprocess(self.ref)
        x = postprocess(self.x_cplx)
        psnr_rec = psnr(ref, x).cpu().numpy()
        ssim_rec = ssim(ref, x, magnitude=False, channel_axis=0).cpu().numpy()
        #psnr_abs_rec = psnr_abs(self.ref, self.x_cplx).item()

        self.metrics['psnr'][self.iter] = psnr_rec
        self.metrics['ssim'][self.iter] = ssim_rec
        #self.metrics['psnr_abs'][self.iter] = psnr_abs_rec
        return
    
    def _get_noise_schedule(self):
        self.noise_dict={}
        if self.config.noise_schedule_config.type == 'edm':
            num_steps = self.config.noise_schedule_config.num_steps
            sigma_min = self.config.noise_schedule_config.sigma_min
            sigma_max = self.config.noise_schedule_config.sigma_max
            rho = self.config.noise_schedule_config.rho
            step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
            self.noise_schedule = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            self.noise_dict = {'noise_schedule': self.noise_schedule.cpu().numpy()}
                            
            if self.config_debug.save_loss_denoise_multilevel:
                # ind_noise = torch.linspace(0, num_steps-1, self.debug_noise_levels, dtype=int)
                # self.noise_levels = self.noise_schedule[ind_noise][:, None, None, None, None]
                # self.noise_dict['noise_levels_loss'] = self.noise_levels.cpu().numpy()

                self.noise_levels = 2*torch.logspace(-3, 1, self.debug_noise_levels, device=self.device, dtype=torch.float32)
                self.noise_levels = self.noise_levels[:, None, None, None, None]
                self.noise_dict['noise_levels_loss'] = self.noise_levels.cpu().numpy()

        return
    
    def _get_losses(self):

        if self.iter > 0:
            e_obs = self.ReconApp.e_obs
            e_denoise = self.ReconApp.e_denoise
            # loss_obs = torch.linalg.vector_norm(e_obs, ord=2, dim=(1,2,3))  # e_obs is complex, (nbatch, ncoils, nx, ny)
            loss_obs = torch.linalg.vector_norm(e_obs, ord=2, dim=(-1))
            # loss_obs = torch.linalg.vector_norm(e_obs, ord=2) 
            loss_denoise = torch.linalg.vector_norm(e_denoise, ord=2, dim=(1,2,3)) # e_denoise is real, (nbatch, 2, nx, ny)
            self.losses['loss_obs'][self.iter] = loss_obs.cpu().numpy()
            self.losses['loss_denoise'][self.iter] = loss_denoise.cpu().numpy()

        if self.config_debug.save_loss_denoise_multilevel:
            # Denoising loss, with multiple noise levels
            eps = torch.randn([self.debug_noise_levels, self.nbatch, 2, self.nx, self.ny], device=self.device, dtype=torch.float32)
            x_t = self.x[None, ...] + self.noise_levels * eps
            x_t = x_t.view(-1, 2, self.nx, self.ny) # (num_noise_instances*nbatch, 2, nx, ny)

            with torch.no_grad():
                denoised = self.denoiser(x_t, self.noise_levels, None)
                score = (x_t - denoised).view(self.debug_noise_levels, self.nbatch, 2, self.nx, self.ny)
                e_mult_denoise =  score / self.noise_levels - eps

                e_mult_denoise = torch.linalg.vector_norm(e_mult_denoise, dim=(2,3,4)).T # Mean on batch dimension
            loss_denoise_mult = e_mult_denoise

            self.losses['loss_denoise_multilevel'][self.iter] = loss_denoise_mult.cpu().numpy()

        if self.config_debug.save_loss_multifreq:
            multifreq_loss = self._get_multifreq_loss()
            self.losses['loss_multifreq'][self.iter] = multifreq_loss.cpu().numpy()
        return
    
    def _get_gradients(self):
        if self.iter > 0:
            # print(self.ReconApp.x.shape)
            # print(self.gradients['grad_obs'].shape)
            e_obs = self.ReconApp.e_obs
            # print(e_obs.shape)
            self.gradients['grad_obs'][self.iter] = self.ReconApp.Ht(e_obs).view(*self.ReconApp.x.shape).detach().cpu().numpy()
            self.gradients['grad_denoise'][self.iter] = self.ReconApp.e_denoise.view(*self.ReconApp.x.shape).detach().cpu().numpy()
        return

    
    def save_debug_outputs(self, iter):
        self.iter = iter + 1
        
        self.x = self.ReconApp.x.clone().detach()
        self.x_cplx = check_cplx(self.x)

        if self.iter > 0:
            self.weights['weights'][self.iter] = self.ReconApp.weights.cpu().numpy()
            self.nnls_flag['nnls_flag'][self.iter] = self.ReconApp.nnls_flag#.cpu().numpy()
        self._get_metrics()
        self._get_losses()

        if self.config_debug.save_gradients:
            self._get_gradients()
            # print(self.gradients.keys())
        else:
            self.gradients = {}

        if  (self.iter) % self.config_debug.save_every == 0:
            # save in debug directory
            debug_outputs = self.weights | self.losses | self.metrics | self.gradients | self.nnls_flag
            np.savez(self.debug_filename, **debug_outputs)

            if self.config_debug.save_images:
                x = postprocess(self.x_cplx)
                ref = postprocess(self.ref)

                save_image(x, os.path.join(self.debug_dir, 'iterate_{:03d}'.format(self.iter) + '.png'))
                save_image(x-ref, os.path.join(self.debug_dir, 'error_{:03d}'.format(self.iter) + '.png'))
                save_image(10*(x-ref), os.path.join(self.debug_dir, 'error10x_{:03d}'.format(self.iter) + '.png'))

            if self.config_debug.save_data:
                debug_outputs = {'x': self.x_cplx.cpu().numpy(),
                                 'ref': self.ref.cpu().numpy()}
                                    
                np.savez(os.path.join(self.debug_dir, 'iterate_{:04d}'.format(self.iter) + '.npz'), **debug_outputs)
            
        return
    
class Outputs:
    def __init__(self, config):

        self.config = config
        self.config_noise = config.noise_schedule_config
        self.config_data = config.data_config
        self.config_output = config.output_config
        self.config_optimizer = config.optimizer_config if 'optimizer_config' in config else None  # OptDiff only
        self.save_every = config.debug_config.save_every
        self.output_data = {}
        self.output_dir = None

        # # noise schedule config
        # self.num_steps = self.config_noise.num_steps
        # self.sigma_max = self.config_noise.sigma_max
        # self.sigma_min = self.config_noise.sigma_min
        # self.rho = self.config_noise.rho

        # metrics
        self.psnr_rec = None
        self.psnr_abs_rec = None
        self.ssim_rec = None
        self.lpips_rec = None

        self.lpips_metric = LPIPSMetric(device = torch.device(config.model_config.device))
        self._prepare_output_dir()
        self._get_string_added_to_fname()
        return

    def _include_list(self, keys):
        # The optimizer and lamda entries describe OptDiff; a baseline is described by its own entry. The
        # noise_schedule entry gives the EDM schedule: a baseline on its standard (uniform) schedule leaves it out.
        if 'algo' not in self.config or self.config.algo == 'OptDiff':
            return keys
        drop = ('optimizer', 'lamda') if self.config_noise.type == 'edm' else ('noise_schedule', 'optimizer', 'lamda')
        return [key for key in keys if key not in drop] + ['baseline']

    def _get_string_added_to_fname(self):

        include_list = self._include_list(self.config_output.include_in_filenames)
        include_str = ''

        for key in include_list:
            if key == 'data':
                slice_idx = self.config_data.slice

                include_str += '__DATA__slice_{:03d}'.format(slice_idx)
            
            elif key == 'noise_schedule': 
                num_steps = self.config_noise.num_steps
                sigma_max = self.config_noise.sigma_max
                sigma_min = self.config_noise.sigma_min
                rho = self.config_noise.rho
                include_str += '__NOISESCH__nsteps_{:04d}__smax_{:.2e}__smin_{:.2e}__rho_{:1.0f}'.format(num_steps, sigma_max, sigma_min, rho)

            elif key == 'optimizer':
                momentum = self.config_optimizer.momentum
                optimizer_order = self.config_optimizer.order
                precondition = self.config_optimizer.preconditioner
                init = self.config_optimizer.initial_estimate
                weights = self.config_optimizer.weights
                noise_instances = self.config_optimizer.num_noise_instances
                num_steps = self.config_optimizer.num_steps

                include_str += '__OPT__ord_{:1d}'.format(optimizer_order)
                if momentum:
                    include_str += '__mom'.format(momentum)
                if precondition:
                    include_str += '__precond'.format(precondition)
                include_str += '__init_{:s}__weights_{:s}__ninst_{:02d}__nsteps_{:03d}'.format(init, weights, noise_instances, num_steps)

            elif key == "lamda":
                lamda = self.config_optimizer.lamda
                if lamda is not None:
                    include_str += '__lamda_{:.3f}'.format(lamda)

        self.add_to_fname = include_str
        return 
    
    def _get_strs_txtfile(self, timing=None):
        txt_list = self._include_list(self.config_output.include_in_txt)

        title_line = ''
        data_line = ''

        for key in txt_list:
            if key == 'data':
                slice_idx = self.config_data.slice

                title_line += 'DATA (slice), '
                data_line += '({:02d} ), '.format(slice_idx)
            
            elif key == 'noise_schedule': 
                num_steps = self.config_noise.num_steps
                sigma_max = self.config_noise.sigma_max
                sigma_min = self.config_noise.sigma_min
                rho = self.config_noise.rho

                title_line += 'NOISE SCH (nsteps, smax, smin, rho), '
                data_line += '( {:04d}, {:.2e}, {:.2e}, {:1.0f} ), '.format(num_steps, sigma_max, sigma_min, rho)

            elif key == 'optimizer':
                momentum = self.config_optimizer.momentum
                optimizer_order = self.config_optimizer.order
                precondition = self.config_optimizer.preconditioner
                init = self.config_optimizer.initial_estimate
                weights = self.config_optimizer.weights
                noise_instances = self.config_optimizer.num_noise_instances
                num_steps = self.config_optimizer.num_steps

                title_line += 'OPTIMIZER (order, momentum, preconditioner, initial_estimate, weights, num_noise_instances, nsteps), '
                data_line += '( {:1d}, {:s}, {:s}, {:s}, {:s}, {:02d}, {:03d} ), '.format(optimizer_order, str(momentum), str(precondition), init, weights, noise_instances, num_steps)
            
            elif key == "lamda":
                lamda = self.config_optimizer.lamda
                if lamda is not None:
                    title_line += 'Lambda_hat, '
                    data_line += '{:.3f}, '.format(lamda)

            elif key == 'baseline':
                algo = self.config.algo
                num_steps = self.config_noise.num_steps
                config_baseline = self.config.baseline_config

                title_line += 'BASELINE (algo, nsteps, ' + ', '.join(config_baseline.keys()) + '), '
                data_line += '( {:s}, {:04d}, '.format(algo, num_steps) + ', '.join(str(v) for v in config_baseline.values()) + ' ), '

        title_line += 'METRICS (psnr, ssim, lpips)'
        data_line += '( {:06.3f}, {:.5f}, {:.5f} )'.format(self.psnr_rec, self.ssim_rec, self.lpips_rec)
        # Reconstruction time is only written when requested, to keep the metrics.txt format unchanged otherwise
        if timing and getattr(self.config_output, 'include_timing', False):
            title_line += ', TIME (sec)'
            data_line += ', ( {:.4f} )'.format(timing)
        return title_line, data_line
    
    def _prepare_output_dir(self):

        slice_idx = self.config_data.slice
        output_dir = self.config_output.output_dir
        output_str = self.config_output.output_dir_str

        if self.config_output.create_new_dir:
            self.output_dir = os.path.join(output_dir,
                                    output_str,
                                    'slice_{:d}'.format(slice_idx),
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = output_dir

        return
        

    def load_output_data(self, output_data, mask=None):
        # Images
        for key in output_data:
            self.output_data[key] = check_cplx(output_data[key])

        if mask is not None:
            nkx, nky = mask.shape[-2:]
            nkmax = max(nkx, nky)

            mask_img = mask.clone().detach()

            if nkx == 1:
                # Repeat nkmax that dimension
                mask_img = mask_img.repeat(1, 1, nkmax, 1)
            elif nky == 1:
                # Repeat nkmax that dimension
                mask_img = mask_img.repeat(1, 1, 1, nkmax)
            self.output_data['mask'] = mask_img

        return 
    
    def estimate_metrics(self):
        ref = self.output_data['ref']
        recon = self.output_data['recon']
        self.psnr_rec = psnr(ref, recon, scale=1.).item()
        self.ssim_rec = ssim(ref, recon, magnitude=False, channel_axis=0).item()
        self.lpips_rec = self.lpips_metric(recon, ref)
        # self.psnr_abs_rec = psnr_abs(ref, recon).item()

        return self.psnr_rec, self.ssim_rec, self.lpips_rec

    def save_outputs(self, iter=-1, timing=None):
        # Images
        if self.config_output.save_images:# and (iter+1) % self.save_every == 0:

            for key in self.output_data:
                if key == "mask":
                    continue
                fname = key 
                # if key == 'recon' or key == 'error' or key == 'error5x':
                #     fname += self.add_to_fname[:27]
                # else:
                fname += self.add_to_fname[:27]# should be enough to keep the full number index 
                save_image(self.output_data[key], os.path.join(self.output_dir, fname + '.png'))

        # Data
        if self.config_output.save_data:
            # copy data from output data using for loop iteration
            output_data = {key: self.output_data[key].cpu().numpy() for key in self.output_data}
            # Save self.ouput_data dictionary in a npz file
            np.savez(os.path.join(self.output_dir, self.add_to_fname[2:] + '.npz'), **output_data)

        # Metrics
        if self.config_output.save_txt:
            metrics_file = os.path.join(self.output_dir, 'metrics.txt')

            if not os.path.exists(metrics_file):
                title_line, _ = self._get_strs_txtfile(timing=timing)
                with open(metrics_file, 'w') as file:
                    file.write("Config file: " + str(config_to_log(self.config)) + "\n\n")
                    file.write(title_line + "\n")
            
            _, metrics_txt = self._get_strs_txtfile(timing=timing)
            with open(metrics_file, 'a') as file:
                file.write(metrics_txt + "\n")

        return 

    def add_end_message(self, messages):
        with open(os.path.join(self.output_dir, 'metrics.txt'), 'a') as file:
            for message in messages:
                file.write(message + "\n")
        return

def config_to_log(config):
    """The run's config with every value resolved."""
    return OmegaConf.to_container(config, resolve=True)

def prepare_output_data(recon, ref, y, init=None):
    """Images saved and evaluated for one reconstruction, mapped to [0, 1].

    recon, ref and init (the starting estimate, if the method has one) are in [-1, 1]; y is the measurement.
    The keys are those of FirstOrder.prepare_output.
    """
    output_dict = {}
    ref = postprocess(ref.detach())
    recon = postprocess(recon.detach())
    output_dict['recon'] = recon
    if init is not None:
        output_dict['init'] = postprocess(init.detach())

    dim = y.shape[1] / (recon.shape[0] * 3)
    try:
        output_dict['measurement'] = postprocess(y.view(y.shape[0], 3, int(dim**0.5), int(dim**0.5)))
    except RuntimeError:
        pass  # the measurement is not a square image (e.g. inpainting keeps only the observed pixels)

    output_dict['ref'] = ref
    output_dict['error'] = recon - ref
    output_dict['error5x'] = 5 * (recon - ref)
    output_dict['error10x'] = 10 * (recon - ref)
    return output_dict

def save_image(image, dest_path):
    image = image.squeeze()
    if image.shape[0] == 3:
        image = image.permute(1,2,0) # H,W,3
    # image = torch.abs(image)
    image = (image * 255).clip(0, 255).to(torch.uint8) 
    image = image.cpu().numpy()
    PIL.Image.fromarray(image).save(dest_path)
    return

def save_data(data, dest_path):
    with h5py.File(dest_path, 'w' ) as f:
        f.create_dataset('data', data = data)
    return

def check_cplx(data):
    # if not data.dtype == torch.complex64:
    #     if data.dtype == torch.float32 and data.shape[1] == 2:
    #         return to_cplx(data)
    return data
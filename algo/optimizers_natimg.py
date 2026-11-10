import torch
from scipy.optimize import nnls
from hydra.utils import to_absolute_path
import numpy as np 
from utils.diffusion_natimg import *

class FirstOrder():
    def __init__(self, y, H, denoiser, config, weights_config=None, ref=None, reseed=True):
        # reseed: restart the random generator at config.seed. main_natimg seeds each image itself before
        # drawing its measurement and passes False, so that the run continues that image's random stream.
        self.H = H.H
        self.Ht = H.Ht
        self.H_pinv = H.H_pinv
        self.normal_op = lambda x: self.Ht(self.H(x)).view(*x.shape)


        self.denoiser = denoiser
        self.weights_config = weights_config
        self.config = config
        self.y = y
        self.num_noise_instances = config.num_noise_instances
        self.momentum = config.momentum
        self.preconditioner = config.preconditioner

        self.num_steps = config.num_steps

        self.nnls_flag = False

        if self.preconditioner:
            degree = config.preconditioner_degree
            npz_file = np.load(to_absolute_path(config.preconditioner_file))
            self.precond_coeffs = npz_file['coeff_l2'][degree-1, :degree+1]


        if reseed:
            torch.manual_seed(config.seed)
        self.device = y.device

        self.ref = ref
        if self.config.initial_estimate == 'zero':
            x0 = torch.zeros_like(ref)
        elif self.config.initial_estimate == 'random':
            x0 = torch.randn_like(ref)
        elif self.config.initial_estimate == 'cg_recon':
            x0 = H.H_pinv(y).view(*ref.size())
        elif self.config.initial_estimate == 'adjoint':
            x0 = self.Ht(y).view(*ref.size())
            # x0 = (y).view(*ref.size())

        self.x0 = x0.detach().clone().requires_grad_(False)
        self.x = x0.detach().clone().requires_grad_(False)

        if self.momentum:
            self.z = torch.zeros_like(x0)

        self.nbatch, _, self.nx, self.ny = x0.shape

        if config.weights == "optimal":
            self._estimate_weights = self._weights_optimal
        elif config.weights == "heuristics":
            self._set_heuristics_weights()
            self._estimate_weights = self._weights_heuristics
            

        self.normalize_obs = getattr(config, 'normalize_obs', False)  # divide the data-term weight by ||Hx - y|| (heuristic weights, and optimal weights with a fixed lamda)

        self.iter = 0
        return

    def _set_heuristics_weights(self):

        nweights = 2
        xdata = np.arange(self.num_steps)
        weights_dic = {"alpha": None, "lamda": None}

        if self.momentum:
            nweights += 1
            weights_dic["beta"] = None

        for key in weights_dic.keys():
            func = getattr(self.weights_config, key).function
            popt = getattr(self.weights_config, key).parameters

            if func == 'linear':
                weights_dic[key] = popt[0] * xdata + popt[1]
            elif func == 'linear_2':
                # Two linear pieces, switching at step idx
                p1, idx, p2 = popt
                line1 = p1[0] * xdata[:idx] + p1[1]
                line2 = p2[0] * xdata[idx:] + p2[1]
                weights_dic[key] = np.concatenate([line1, line2])
            elif func == 'exp':
                weights_dic[key] = popt[0] * np.exp(-popt[1] * xdata) + popt[2]
            elif func == 'constant':
                weights_dic[key] = np.ones_like(xdata) * popt
            else:
                raise ValueError(f'Invalid heuristics function {func} for {key}')

            weights_dic[key] = np.maximum(weights_dic[key], 0) # Non-negative weights

        self.weights_heur = np.zeros((nweights, self.num_steps), dtype=np.float32)
        self.weights_heur[0] = weights_dic['alpha']
        self.weights_heur[1] = weights_dic['lamda'] 
        

        if self.momentum:
            self.weights_heur[2] = weights_dic['beta']
            self.weights_heur[2, :2] = 0 # First two iterations with no momentum for stability
        return 
    
    def estimate_update(self, noise_std, t): # t is needed for pretrained reddif diffusion model
        
        self.iter += 1
        t = torch.ones(self.nbatch).to(self.device).long() * t

        # Denoising error
        eps = torch.randn([self.num_noise_instances, self.nbatch, 3, self.nx, self.ny], device=self.device, dtype=self.x.dtype)

        with torch.no_grad():
            x_t = noise_std.sqrt() * self.x[None,...] + (1 - noise_std).sqrt() * eps
            x_t = x_t.view(-1, 3, self.nx, self.ny) # (num_noise_instances*nbatch, 3, nx, ny)
            et = self.denoiser(x_t, t)[:, :3]
            e_denoise = et - eps
            
            self.e_denoise = torch.mean(e_denoise, axis=0) # Mean on instances dimension
            grad_denoise = self.e_denoise
        
        # Observational error
        self.e_obs = self.H(self.x) - self.y
        grad_obs = self.Ht(self.e_obs)
        grad_obs = grad_obs.view(*grad_denoise.shape)

        if self.preconditioner:
            grad_obs = self.preconditioning(self.precond_coeffs, grad_obs)


        updates = torch.stack([grad_obs, grad_denoise], dim=0)

        if self.momentum:
            updates = torch.concat([updates, self.z[None, ...]], dim=0)

        self.updates = updates

        self._estimate_weights()

        return 
    
    def estimate_update_cnst_lamda(self, noise_std, t, lamda): # t is needed for pretrained reddif diffusion model
        
        self.iter += 1
        t = torch.ones(self.nbatch).to(self.device).long() * t

        # Denoising error
        eps = torch.randn([self.num_noise_instances, self.nbatch, 3, self.nx, self.ny], device=self.device, dtype=self.x.dtype)

        with torch.no_grad():
            x_t = noise_std.sqrt() * self.x[None,...] + (1 - noise_std).sqrt() * eps
            x_t = x_t.view(-1, 3, self.nx, self.ny) # (num_noise_instances*nbatch, 3, nx, ny)
            et = self.denoiser(x_t, t)[:, :3]
            e_denoise = et - eps
            
            self.e_denoise = torch.mean(e_denoise, axis=0) # Mean on instances dimension
            grad_denoise = self.e_denoise
            # loss_denoise = torch.mul(e_denoise.detach(), mu_x).sum()
        
        # Observational error
        self.e_obs = self.H(self.x) - self.y
        grad_obs = self.Ht(self.e_obs)
        grad_obs = grad_obs.view(*grad_denoise.shape)

        if self.preconditioner:
            grad_obs = self.preconditioning(self.precond_coeffs, grad_obs)


        loss_denoise = torch.linalg.vector_norm(self.e_denoise, ord=2, dim=(1,2,3))
        loss_obs = torch.linalg.vector_norm(self.e_obs, ord=2, dim=(-1))
        # normalize_obs: divide the data term by ||Hx - y||
        obs_norm = loss_obs + 1e-8 if self.normalize_obs else torch.ones_like(loss_obs)
        updates = grad_obs / obs_norm[:, None, None, None] + lamda * grad_denoise / loss_denoise[:, None, None, None]


        if self.momentum:
            updates = torch.concat([updates[None, ...], self.z[None, ...]], dim=0)
            # print(updates.shape) # [2, 1, 3, 256, 256]

        ref_dx = self.x - (self.ref)
        # Jointly estimated
        b_w = ref_dx.flatten().detach().cpu().numpy()

        # flatten all elements in updates
        A_w = updates.flatten(start_dim=1).T 
        A_w = A_w.detach().cpu().numpy()
        
        # Non-negative least-squares: alpha_hat (and beta with momentum)
        weights = nnls(A_w, b_w)[0]
        self.nnls_flag = (weights[0] == 0)

        if self.momentum:
            self.weights = torch.tensor([weights[0] / obs_norm.squeeze(), lamda * weights[0] / loss_denoise.squeeze(), weights[1]], device=self.device, dtype=self.x.dtype)
            self.updates = torch.stack([grad_obs, grad_denoise, self.z], dim=0)
        else:
            self.weights = torch.tensor([weights[0] / obs_norm.squeeze(), lamda * weights[0] / loss_denoise.squeeze()], device=self.device, dtype=self.x.dtype)
            self.updates = torch.stack([grad_obs, grad_denoise], dim=0)

        return 
    
    def step(self):
        update = torch.sum(self.weights[:, None, None, None, None] * self.updates, dim=0)
        self.x -= update

        if self.momentum:
            self.z = update
        return
    
    
    def prepare_output(self, ref=None):

        output_dict = {}
        ref = postprocess(ref.detach()) # postprocessing for natural image task
        recon = postprocess(self.x.detach())
        output_dict['recon'] = recon 
        output_dict['init'] = postprocess(self.x0.detach())
        # print(recon.shape, ref.shape)
        dim = self.y.shape[1] / (recon.shape[0] * 3)
        # print(dim)
        
        try:
            output_dict['measurement'] = postprocess(self.y.view(self.y.shape[0], 3, int(dim**0.5), int(dim**0.5)))
        except:
            pass
        
        if ref is not None:
            output_dict['ref'] = ref
            output_dict['error'] = recon - ref
            output_dict['error5x'] = 5 * (recon - ref)
            output_dict['error10x'] = 10 * (recon - ref)

        return output_dict

    def preconditioning(self, c, x):
        x1 = x.detach().clone()
        xp = torch.zeros_like(x)
        for i in range(len(c)):
            xp += c[i] * x1
            x1 = self.normal_op(x1)
        return xp
    
    def _weights_optimal(self):

        ref_dx = self.x - (self.ref) 
        # Jointly estimated
        b_w = ref_dx.flatten().detach().cpu().numpy()

        # flatten all elements in updates
        A_w = self.updates.flatten(start_dim=1).T 
        A_w = A_w.detach().cpu().numpy()
        
        # Non-negative least-squares
        self.weights = nnls(A_w, b_w)[0]
        self.weights = torch.tensor(self.weights, device=self.device, dtype=self.x.dtype)

        return

    def _weights_heuristics(self):
        # print(self.e_obs.shape)
        loss_obs = torch.linalg.vector_norm(self.e_obs.flatten(), ord=2)  # e_obs 
        loss_denoise = torch.linalg.vector_norm(self.e_denoise, ord=2, dim=(1,2,3)) # e_denoise 
        

        # Lamda is also weighted by the denoising error norm
        self.weights = torch.from_numpy(self.weights_heur[:,self.iter - 1]).to(self.device)
        if self.weights[0] > 0:
            self.weights[1] = self.weights[1]/ (loss_denoise + 1e-8) * self.weights[0] 
            
        else:
            self.weights[1] = self.weights[1]/ (loss_denoise + 1e-8)

        if self.normalize_obs:
            self.weights[0] = self.weights[0] / (loss_obs + 1e-8)
        return

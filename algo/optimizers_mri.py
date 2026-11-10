import numpy as np
import torch
from scipy.optimize import nnls

from utils.cg import ConjugateGradient
from utils.data_mri import to_real


class FirstOrder():
    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None):
        self.forward_op = operators['forward']
        self.adjoint_op = operators['adjoint']
        self.normal_op = lambda x: self.adjoint_op(self.forward_op(x))

        self.denoiser = denoiser
        self.weights_config = weights_config
        self.config = config
        self.y = y
        self.num_noise_instances = config.num_noise_instances
        self.momentum = config.momentum
        self.preconditioner = config.preconditioner

        self.num_steps = config.num_steps

        if self.preconditioner:
            degree = config.preconditioner_degree
            npz_file = np.load(config.preconditioner_file)
            self.precond_coeffs = npz_file['coeff_l2'][degree-1, :degree+1]

        torch.manual_seed(config.seed)
        self.device = y.device
        self.x0 = None
        self.ref = ref

        if config.weights == "optimal":
            self._estimate_weights = self._weights_optimal
        elif config.weights == "optimal_cnst_lamda":
            self.lambda_hat = config.reddiff_lambda
            self._estimate_weights = self._weights_optimal_cnst_lamda
        elif config.weights == "heuristics":
            self._set_heuristics_weights()
            self._estimate_weights = self._weights_heuristics
        elif config.weights == "reddiff":
            if self.momentum:
                raise ValueError('reddiff weights do not support momentum.')
            self._estimate_weights = self._weights_reddiff
            self.reddiff_lambda = config.reddiff_lambda
            self.reddiff_weight_fn = config.reddiff_weight_fn
            self.reddiff_do_optimal_step = getattr(config, 'reddiff_do_optimal_step', False)
        else:
            raise ValueError(f'Invalid weights method: {config.weights}')
            
        self._initial_estimate()
        self.iter = 0
        return

    def _initial_estimate(self):
        x0 = self.adjoint_op(self.y)

        if self.config.initial_estimate == 'zero':
            x0 = torch.zeros_like(x0)
        elif self.config.initial_estimate == 'random':
            x0 = torch.randn_like(x0)
        elif self.config.initial_estimate == 'adjoint':
            pass
        elif self.config.initial_estimate == 'cg_recon':
            num_cg_iter = 30
            lambda_cg = 1e-3
            model_normal = lambda x: self.adjoint_op(self.forward_op(x)) + lambda_cg * x
            cg_solve = ConjugateGradient(model_normal, num_cg_iter, dbprint=False)
            x0 = cg_solve(x0, x0)
        else:
            raise ValueError('Invalid initial estimate method')
        self.x0 = x0.detach().clone().requires_grad_(False)
        self.x = x0.detach().clone().requires_grad_(False)

        if self.momentum:
            self.z = x0.detach().clone().requires_grad_(False)
        self.nbatch, _, self.nx, self.ny = x0.shape
        return 
    
    def _set_heuristics_weights(self):
        # 'raw': the fitted curves give the weights directly (w_obs = alpha, w_denoise = lamda * alpha, w_mom = beta).
        # 'normalized': OptDiff reparameterization with alpha hat, lamda hat and beta hat; at every step the observation
        # weight is divided by ||e_obs|| and the denoiser weight by ||grad_denoise|| (see _weights_heuristics), and the
        # momentum weight is beta * alpha.
        self.parameterization = getattr(self.weights_config, 'parameterization', 'raw')
        if self.parameterization not in ('raw', 'normalized'):
            raise ValueError(f'Invalid weights parameterization: {self.parameterization}')
        
        if hasattr(self.weights_config, 'file'):
            if self.parameterization != 'raw':
                raise ValueError('Weights loaded from a file must use the raw parameterization.')
            self.weights_heur = np.load(self.weights_config.file)
            # self.weights_heur[2, :1] = 0 # stability
        else:
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
                elif func == 'constant':
                    weights_dic[key] = np.ones_like(xdata) * popt
                elif func == 'linear_2':
                    # Two linear pieces, switching at step idx
                    p1, idx, p2 = popt
                    line1 = p1[0] * xdata[:idx] + p1[1]
                    line2 = p2[0] * xdata[idx:] + p2[1]
                    weights_dic[key] = np.concatenate([line1, line2])
                elif func == 'exp':
                    weights_dic[key] = popt[0] * np.exp(-popt[1] * xdata) + popt[2]
                else:
                    raise ValueError(f'Invalid heuristics function {func} for {key}')

            self.weights_heur = np.zeros((nweights, self.num_steps), dtype=np.float32)
            self.weights_heur[0] = weights_dic['alpha']
            self.weights_heur[1] = weights_dic['lamda'] * weights_dic['alpha']

            if self.momentum:
                if self.parameterization == 'normalized':
                    self.weights_heur[2] = weights_dic['beta'] * weights_dic['alpha']
                else:
                    self.weights_heur[2] = weights_dic['beta']
                self.weights_heur[2, :2] = 0 # First two iterations with no momentum for stability
        return 
    
    def estimate_update(self, noise_std):
        
        self.iter += 1

        # Denoising error
        eps = torch.randn([self.num_noise_instances, self.nbatch, 2, self.nx, self.ny], device=self.device, dtype=self.x.dtype)

        with torch.no_grad():
            x_t = self.x[None,...] + noise_std * eps # (num_noise_instances, nbatch, 2, nx, ny)
            x_t = x_t.view(-1, 2, self.nx, self.ny) # (num_noise_instances*nbatch, 2, nx, ny)
            denoised = self.denoiser(x_t, noise_std, None)#.to(torch.float64)
            score = (x_t - denoised).view(self.num_noise_instances, self.nbatch, 2, self.nx, self.ny)
            e_denoise =  score / noise_std - eps
            
            self.e_denoise = torch.mean(e_denoise, axis=0) # Mean on instances dimension
            grad_denoise = self.e_denoise
            # loss_denoise = torch.mul(e_denoise.detach(), mu_x).sum()
        
        # Observational error
        self.e_obs = self.forward_op(self.x) - self.y
        grad_obs = self.adjoint_op(self.e_obs)

        if self.preconditioner:
            grad_obs = self.preconditioning(self.precond_coeffs, grad_obs)

        updates = torch.stack([grad_obs, grad_denoise], dim=0)

        if self.momentum:
            updates = torch.concat([updates, self.z[None, ...]], dim=0)

        self.updates = updates

        self._estimate_weights(noise_std=noise_std)

        return 
    
    def step(self):
        update = torch.sum(self.weights[:, None, None, None, None] * self.updates, dim=0)
        self.x -= update

        if self.momentum:
            self.z = update
        return
    
    def prepare_output(self, ref=None):

        output_dict = {}
        output_dict['recon'] = self.x.detach()
        output_dict['init'] = self.x0.detach()

        if ref is not None:
            output_dict['ref'] = ref.detach()
            output_dict['error'] = self.x - ref
            output_dict['error5x'] = 5 * (self.x - ref)

        return output_dict

    def preconditioning(self, c, x):
        x1 = x.detach().clone()
        xp = torch.zeros_like(x)
        for i in range(len(c)):
            xp += c[i] * x1
            x1 = self.normal_op(x1)
        return xp
    
    def _weights_optimal(self, **kwargs):

        ref_dx = self.x - to_real(self.ref) 
        # Jointly estimated
        b_w = ref_dx.flatten().detach().cpu().numpy()

        # flatten all elements in updates
        A_w = self.updates.flatten(start_dim=1).T 
        A_w = A_w.detach().cpu().numpy()
        
        # Non-negative least-squares
        self.weights = nnls(A_w, b_w)[0]
        self.weights = torch.tensor(self.weights, device=self.device, dtype=self.x.dtype)

        return 
    
    def _weights_optimal_cnst_lamda(self, **kwargs):
        # Normalized direction grad_obs / ||e_obs|| + lamda_hat * grad_denoise / ||grad_denoise|| with a constant lamda_hat.
        # Only the step size (and the momentum weight) are estimated with NNLS against the reference.
        grad_obs = self.updates[0]
        grad_denoise = self.updates[1]
        loss_obs = torch.linalg.vector_norm(self.e_obs, ord=2, dim=(1,2,3))  # e_obs is complex, (nbatch, ncoils, nx, ny)
        loss_denoise = torch.linalg.vector_norm(self.e_denoise, ord=2, dim=(1,2,3)) # e_denoise is real, (nbatch, 2, nx, ny)

        direction = grad_obs / loss_obs[:, None, None, None] + self.lambda_hat * grad_denoise / loss_denoise[:, None, None, None]
        if self.momentum:
            directions = torch.stack([direction, self.updates[2]], dim=0)
        else:
            directions = direction[None, ...]

        ref_dx = self.x - to_real(self.ref)
        b_w = ref_dx.flatten().detach().cpu().numpy()
        A_w = directions.flatten(start_dim=1).T
        A_w = A_w.detach().cpu().numpy()

        # Non-negative least-squares
        weights = nnls(A_w, b_w)[0]
        weights = torch.tensor(weights, device=self.device, dtype=self.x.dtype)

        # Weights of grad_obs, grad_denoise (and momentum), applied to self.updates in step()
        if self.momentum:
            self.weights = torch.tensor([weights[0] / loss_obs.squeeze(), self.lambda_hat * weights[0] / loss_denoise.squeeze(), weights[1]], device=self.device, dtype=self.x.dtype)
        else:
            self.weights = torch.tensor([weights[0] / loss_obs.squeeze(), self.lambda_hat * weights[0] / loss_denoise.squeeze()], device=self.device, dtype=self.x.dtype)

        return

    def _weights_heuristics(self, **kwargs):

        self.weights = torch.from_numpy(self.weights_heur[:,self.iter - 1]).to(self.device)
        if self.parameterization == 'normalized':
            # alpha hat and lamda hat are relative to the norms of the observation and denoising errors
            loss_obs = torch.linalg.vector_norm(self.e_obs, ord=2, dim=(1,2,3))  # e_obs is complex, (nbatch, ncoils, nx, ny)
            loss_denoise = torch.linalg.vector_norm(self.e_denoise, ord=2, dim=(1,2,3)) # e_denoise is real, (nbatch, 2, nx, ny)
            self.weights[0] = self.weights[0] / loss_obs
            self.weights[1] = self.weights[1] / loss_denoise
        return

    def _weights_reddiff(self, **kwargs):

        snr_inv = kwargs['noise_std']

        weight_fn = self.reddiff_weight_fn
        if weight_fn == 'linear':
            weight_t = snr_inv
        elif weight_fn == 'sqrt':
            weight_t = torch.sqrt(snr_inv)
        elif weight_fn == 'log':
            weight_t = torch.log(1+snr_inv)
        elif weight_fn == 'square':
            weight_t = snr_inv**2
        elif weight_fn == 'constant':
            weight_t = 1.0
        else:
            raise ValueError('Invalid reddiff weighting function')

        lambda_t = self.reddiff_lambda * weight_t

        if self.reddiff_do_optimal_step:

            ref_dx = self.x - to_real(self.ref)
            # Jointly estimated
            b_w = ref_dx.flatten().detach().cpu().numpy()

            # flatten all elements in updates
            modified_update = self.updates[0] + self.updates[1] * lambda_t
            A_w = modified_update.flatten(start_dim=1).T
            A_w = A_w.detach().cpu().numpy()

            # Non-negative least-squares
            optimal_stepsize = nnls(A_w, b_w)[0][0]
            self.weights = torch.tensor([optimal_stepsize, optimal_stepsize * lambda_t], device=self.device, dtype=self.x.dtype)

        else:
            self.weights = torch.tensor([1., lambda_t], device=self.device, dtype=self.x.dtype)

        return

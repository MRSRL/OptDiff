"""MRI baselines: DPS, PiGDM, DAPS and DiffPIR, run by main_mri.py with algo=<name> (configs/mri/algo)."""
import numpy as np
import torch
import tqdm

from utils.cg import ConjugateGradient
from utils.diffusion_mri import Scheduler, DiffusionSampler


class InverseSampler:
    """
    Minimal base class for configuring samplers/optimizers:
      - __init__(y, operators, denoiser, config, weights_config=None, ref=None)
      - estimate_update(noise_std)
      - step()
      - prepare_output(ref=None)
    """
    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None, noise_schedule_config=None):

        # ops
        self.forward_op = operators['forward']
        self.adjoint_op = operators['adjoint']
        self.normal_op = lambda x: self.adjoint_op(self.forward_op(x))

        # model + config
        self.denoiser = denoiser
        self.config = config
        self.weights_config = weights_config
        self.noise_schedule_config = noise_schedule_config
        self.y = y
        self.device = y.device
        self.dtype  = y.dtype

        # algorithm params
        self.num_steps = config.num_steps
        self.num_noise_instances = config.num_noise_instances
        self.preconditioner = config.preconditioner
        self.momentum = config.momentum

        torch.manual_seed(config.seed)

        # init state
        self.x0 = None
        self.ref = ref
        self._initial_estimate()

        # buffers used by debugger
        self.e_obs = torch.zeros_like(self.y, dtype=self.y.dtype)
        self.e_denoise = torch.zeros_like(self.x, dtype=self.x.dtype)

        self.iter = 0

    def _initial_estimate(self):
        x0 = self.adjoint_op(self.y)

        init = self.config.initial_estimate
        if init == 'zero':
            x0 = torch.zeros_like(x0)
        elif init == 'random':
            x0 = torch.randn_like(x0)
        elif init == 'adjoint':
            pass
        elif init == 'cg_recon':
            num_cg_iter = 30
            lambda_cg = 1e-3
            model_normal = lambda x: self.adjoint_op(self.forward_op(x)) + lambda_cg * x
            cg_solve = ConjugateGradient(model_normal, num_cg_iter, dbprint=False)
            x0 = cg_solve(x0, x0)
        else:
            raise ValueError(f'Invalid initial estimate method: {init}')

        self.x0 = x0.detach().clone().requires_grad_(False)
        self.x  = x0.detach().clone().requires_grad_(False)
        if self.momentum:
            self.z = x0.detach().clone().requires_grad_(False)
        self.nbatch, _, self.nx, self.ny = x0.shape

    def preconditioning(self, c, x):
        x1 = x.detach().clone()
        xp = torch.zeros_like(x)
        for i in range(len(c)):
            xp += c[i] * x1
            x1 = self.normal_op(x1)
        return xp

    def prepare_output(self, ref=None):
        output_dict = {}
        output_dict['recon'] = self.x.detach()
        output_dict['init'] = self.x0.detach()
        if ref is not None:
            output_dict['ref'] = ref.detach()
            output_dict['error'] = self.x - ref
            output_dict['error5x'] = 5 * (self.x - ref)
        return output_dict


class DPS(InverseSampler):
    """
    Diffusion Posterior Sampling (linear Gaussian):
    Uses guidance g_t ~ H^T (y - H x0) with residual-normalized step size:
        zeta_i = lambda_base / || y - H( x0_hat ) ||_2.
    """
    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None):
        # make sure preconditioner and momentum are off, give error if not
        if config.preconditioner:
            raise ValueError('DPS does not support preconditioning.')
        if config.momentum:
            raise ValueError('DPS does not support momentum.')
        super().__init__(y, operators, denoiser, config, weights_config, ref)
        # DPS-specific knobs
        self.lambda_base = float(getattr(config, "dps_lambda", 1.0))
        self.eta = float(getattr(config, "dps_eta", 0.0))  # stochasticity in prior step
        self.alpha_mode = getattr(config, "alpha_from_sigma_mode", "edm")  # 'edm' or 'vp'
        self.sigma_next = None

    def _to_tensor(self, val):
        return torch.tensor(val, device=self.device, dtype=self.x.dtype)

    def set_next_sigma(self, sigma_next):
        """Optional per-step setter for the next sigma used by the DDIM prior step."""
        if sigma_next is None:
            self.sigma_next = None
            return
        self.sigma_next = float(sigma_next) if not torch.is_tensor(sigma_next) else float(sigma_next)

    def _alpha_bar_from_sigma(self, sigma):
        s = float(sigma)
        if self.alpha_mode.lower() == 'vp':
            ab = 1.0 - s**2
        elif self.alpha_mode.lower() == 'edm': # EDM and VE
            ab = 1.0 / (1.0 + s**2)
        else:
            raise ValueError(f"Unknown alpha_from_sigma_mode: {self.alpha_mode}")
        return self._to_tensor(max(1e-6, min(1.0 - 1e-6, ab)))

    def _zeta_from_residual(self, res, eps=1e-12):
        # per-sample L2 norm over non-batch dims
        b = res.shape[0]
        denom = res.reshape(b, -1).norm(dim=1).clamp_min(eps)
        zeta = self.lambda_base / denom  # shape [B]
        # reshape for broadcasting to [B, C, H, W]
        return zeta.view(b, 1, 1, 1)

    def estimate_update(self, noise_std):
        self.iter += 1

        # alpha bars
        ab_i   = self._alpha_bar_from_sigma(noise_std)
        ab_im1 = self._alpha_bar_from_sigma(self.sigma_next)  if (self.sigma_next is not None) else ab_i

        # calculate c1 and c2 for DDIM step
        c1 = torch.sqrt((1 - ab_i / ab_im1) * (1 - ab_im1) / (1 - ab_i)) * self.eta
        c2 = torch.sqrt((1 - ab_im1) - c1 ** 2)

        # prepare x_i for backprop through denoiser
        x_i = self.x.detach().clone().requires_grad_(True)

        # apply DDIM step to get x_{i-1}_prime
        x0_hat = self.denoiser(x_i, noise_std, None)
        eps_hat = 1/torch.sqrt(1 - ab_i) * (x_i - torch.sqrt(ab_i) * x0_hat)
        eps = torch.randn([self.nbatch, 2, self.nx, self.ny], device=self.device, dtype=self.x.dtype)
        x_prime = torch.sqrt(ab_im1) * x0_hat + c1 * eps + c2 * eps_hat

        # likelihood: norm(y - H x0_hat)**2 then backprop through x0_hat
        res = self.y - self.forward_op(x0_hat)
        res_norm = res.reshape(self.nbatch, -1).norm(dim=1)
        grad_term = torch.autograd.grad(outputs=(res_norm**2).sum(), inputs=x_i)[0]
        
        # residual norm based grad coefficient
        with torch.no_grad():
            zeta = self._zeta_from_residual(res)
        x_im1 = x_prime - zeta * grad_term
        self.x_im1 = x_im1.detach()

        # update debug buffers
        with torch.no_grad():
            self.e_obs = self.forward_op(self.x) - self.y
            self.e_denoise = self.x - x0_hat.detach()

        # consume sigma_next
        self.sigma_next = None

        return

    def step(self):
        self.x = self.x_im1
        return


class PiGDM(InverseSampler):
    """
    Pseudoinverse-Guided Diffusion Models:
      v = H^\dagger (y - H x0)  (compute via CG solve of normal eqs)
      g = J^T v, where J = d x0 / d x_t (vJP through the denoiser).
      x <- x + lambda_t * g
    """
    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None):
        # make sure preconditioner and momentum are off, give error if not
        if config.preconditioner:
            raise ValueError('PiGDM does not support preconditioning.')
        if config.momentum:
            raise ValueError('PiGDM does not support momentum.')
        super().__init__(y, operators, denoiser, config, weights_config, ref)
        # PiGDM-specific knobs
        self.lambda_base = float(getattr(config, "pigdm_lambda", 1.0))
        self.eta = float(getattr(config, "pigdm_eta", 0.0))  # stochasticity in prior step
        self.alpha_mode = getattr(config, "alpha_from_sigma_mode", "edm")  # 'edm' or 'vp'
        self.sigma_next = None

        self.tikh_reg = float(getattr(config, "pigdm_tikh_reg", 1e-4))     # Tikhonov for pseudo-inverse
        self.cg_iters = int(getattr(config, "pigdm_cg_iters", 30))

    def _to_tensor(self, val):
        return torch.tensor(val, device=self.device, dtype=self.x.dtype)

    def set_next_sigma(self, sigma_next):
        """Optional per-step setter for the next sigma used by the DDIM prior step."""
        if sigma_next is None:
            self.sigma_next = None
            return
        self.sigma_next = float(sigma_next) if not torch.is_tensor(sigma_next) else float(sigma_next)

    def _alpha_bar_from_sigma(self, sigma):
        s = float(sigma)
        if self.alpha_mode.lower() == 'vp':
            ab = 1.0 - s**2
        elif self.alpha_mode.lower() == 'edm': # EDM and VE
            ab = 1.0 / (1.0 + s**2)
        else:
            raise ValueError(f"Unknown alpha_from_sigma_mode: {self.alpha_mode}")
        return self._to_tensor(max(1e-6, min(1.0 - 1e-6, ab)))

    def _zeta_from_residual(self, res, eps=1e-12):
        # per-sample L2 norm over non-batch dims
        b = res.shape[0]
        denom = res.reshape(b, -1).norm(dim=1).clamp_min(eps)
        zeta = self.lambda_base / denom  # shape [B]
        # reshape for broadcasting to [B, C, H, W]
        return zeta.view(b, 1, 1, 1)

    def _apply_pinv_residual(self, residual):
        # Solve (H^T H + eta I) u = H^T residual  -> u ~= H^\dagger residual
        rhs = self.adjoint_op(residual)
        normal = lambda x: self.normal_op(x) + self.tikh_reg * x
        cg_solve = ConjugateGradient(normal, self.cg_iters, dbprint=False)
        u0 = torch.zeros_like(rhs)
        u = cg_solve(u0, rhs)
        return u

    def estimate_update(self, noise_std):

        self.iter += 1

        # alpha bars
        ab_i   = self._alpha_bar_from_sigma(noise_std)
        ab_im1 = self._alpha_bar_from_sigma(self.sigma_next)  if (self.sigma_next is not None) else ab_i

        # calculate c1 and c2 for DDIM step
        c1 = torch.sqrt((1 - ab_i / ab_im1) * (1 - ab_im1) / (1 - ab_i)) * self.eta
        c2 = torch.sqrt((1 - ab_im1) - c1 ** 2)

        # prepare x_i for backprop through denoiser
        x_i = self.x.detach().clone().requires_grad_(True)

        # apply DDIM step to get x_{i-1}_prime
        x0_hat = self.denoiser(x_i, noise_std, None)
        eps_hat = 1/torch.sqrt(1 - ab_i) * (x_i - torch.sqrt(ab_i) * x0_hat)
        eps = torch.randn([self.nbatch, 2, self.nx, self.ny], device=self.device, dtype=self.x.dtype)
        x_prime = torch.sqrt(ab_im1) * x0_hat + c1 * eps + c2 * eps_hat

        # residual and pseudoinverse back-projection
        with torch.no_grad():
            res = self.y - self.forward_op(x0_hat)
            v   = self._apply_pinv_residual(res)     # A^† (y - A x0_hat)
        
        v_x = (v.detach() * x0_hat).sum()  # dot(v, x0_hat)
        # vJP through denoiser: g = J_{x0}(x_i)^T v
        grad_term = torch.autograd.grad(outputs=v_x, inputs=x_i,
                                retain_graph=False, create_graph=False)[0]

        zeta = 1.0 * self.lambda_base

        # finalize step
        x_im1 = x_prime + zeta * grad_term
        self.x_im1 = x_im1.detach()

        # update debug buffers
        with torch.no_grad():
            self.e_obs = self.forward_op(self.x) - self.y
            self.e_denoise = self.x - x0_hat.detach()

        # consume sigma_next
        self.sigma_next = None

        return

    def step(self):
        self.x = self.x_im1
        return


class LangevinDynamics:
    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        self.num_steps = int(num_steps)
        self.lr = float(lr)
        self.tau = float(tau)
        self.lr_min_ratio = float(lr_min_ratio)

    def _lr_at(self, ratio):
        # matches exponential lr
        return self.lr * (self.lr_min_ratio ** ratio)

    def sample(self, x0hat, forward_op, adjoint_op, y, sigma, ratio, verbose=False):
        iters = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)

        x0hat = x0hat.detach()
        x = x0hat.clone()

        for _ in iters:
            # grad = H^T(Hx - y)/τ^2 + (x - x0hat)/σ^2
            r = forward_op(x) - y
            grad = adjoint_op(r) / (self.tau ** 2) + (x - x0hat) / (sigma ** 2)
            x = x - lr * grad + torch.randn_like(x) * np.sqrt(2.0 * lr)

            if torch.isnan(x).any():
                return torch.zeros_like(x)
        return x.detach()
    
    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class DAPS(InverseSampler):
    """
    DAPS (Decoupled Annealing Posterior Sampling) for MRI.
    Outer step at σ_k:
      1) Reverse (probability-flow ODE, Euler) from x_t@σ_k down to ~0 → x0_hat
      2) Langevin dynamics on x0 at fixed σ_k using:
             grad = (H^T(Hx - y))/tau^2  +  (x - x0_hat)/σ_k^2
         x <- x - lr * grad + sqrt(2*lr) * N(0, I)
      3) Forward re-noise: x_t_next = x0_y + σ_{k+1} * ε

    Notes:
      - Uses VE/EDM sigma parameterization; ᾱ is only for helper conversions elsewhere.
      - No vJP through the denoiser in Langevin (matches DAPS repo).
    """

    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None, noise_schedule_config=None):
        if getattr(config, "preconditioner", False):
            raise ValueError("DAPS does not support preconditioning.")
        if getattr(config, "momentum", False):
            raise ValueError("DAPS does not support momentum.")
        super().__init__(y, operators, denoiser, config, weights_config, ref, noise_schedule_config)

        # ---------------- Langevin hyperparams -------------
        self.lg_steps        = config.daps_lgvd_config.num_steps
        self.lg_lr           = config.daps_lgvd_config.lr
        self.lg_lr_min_ratio = config.daps_lgvd_config.lr_min_ratio
        self.tau             = config.daps_lgvd_config.tau

        # ----- State -----
        self.sigma_next = None
        self.xt = None                  # noisy sample at current outer σ
        self.x_im1 = None               # clean estimate after current outer step

        # --------------- Specifics -------------------------
        self.diffusion_scheduler_config = config.daps_diffusion_scheduler_config
        self.lgvd = LangevinDynamics(self.lg_steps, self.lg_lr, self.tau, self.lg_lr_min_ratio)

    def set_next_sigma(self, sigma_next):
        """Optional per-step setter for the next sigma used by annealing scheduler."""
        if sigma_next is None:
            self.sigma_next = None
            return
        self.sigma_next = float(sigma_next) if not torch.is_tensor(sigma_next) else float(sigma_next)

    def estimate_update(self, noise_std):
        self.iter += 1
        sigma_k = float(noise_std)

        # Initialize noisy state the first time we see this schedule
        if (self.xt is None) or (self.iter == 1):
            self.xt = torch.randn(self.nbatch, 2, self.nx, self.ny,
                                device=self.device, dtype=self.x.dtype) * sigma_k

        # 1) reverse diffusion (PF-ODE Euler) to x0hat
        xt = self.xt.detach().clone()
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma_k)
        sampler = DiffusionSampler(diffusion_scheduler)
        x0hat = sampler.sample(self.denoiser, xt, SDE=False, verbose=False)

        # 2) Langevin dynamics on x0 at fixed σ_k
        x0y = self.lgvd.sample(
            x0hat, self.forward_op, self.adjoint_op, self.y,
            sigma=sigma_k, ratio=(self.iter - 1) / self.noise_schedule_config.num_steps
        )
        self.x0y = x0y.detach()

        # 3) forward re-noise for next outer level (if provided)
        if (self.sigma_next is not None) and (self.sigma_next > 0):
            self.xt = (x0y + torch.randn_like(x0y) * float(self.sigma_next)).detach()
        else:
            self.xt = None  # last step

        # optional debug
        with torch.no_grad():
            self.e_obs = self.forward_op(self.x0y) - self.y
            self.e_denoise = self.x0y - x0hat

        self.sigma_next = None


    def step(self):
        self.x = self.x0y
        return


class DiffPIR(InverseSampler):
    """
    DiffPIR: Denoising Diffusion Models for Plug-and-Play Image Restoration.
    HQS-based method that couples a DDIM prior step with a data proximal subproblem:
      At each step (sigma_i -> sigma_next):
        1) Denoise: x0 = D(xt, sigma_i)
        2) Data proximal (HQS): x0hat = argmin_x ||Ax-y||^2 + rho_i * ||x - x0||^2
               rho_i = 2 * lambda * sigma_n^2 / sigma_i^2
           Gradient step (default): x0hat = x0 - A^T(Ax0 - y) / rho_i
           CG solve (optional):     (A^T A + rho_i I) x0hat = A^T y + rho_i * x0
        3) DDIM step: xt_next = x0hat + (sqrt(xi)*eps + sqrt(1-xi)*effect) * sigma_next
               effect = (xt - x0hat) / sigma_i   (predicted noise direction)
    xi=0 gives deterministic DDIM; xi in (0,1] adds stochasticity.
    """

    def __init__(self, y, operators, denoiser, config, weights_config=None, ref=None):
        if getattr(config, "preconditioner", False):
            raise ValueError('DiffPIR does not support preconditioning.')
        if getattr(config, "momentum", False):
            raise ValueError('DiffPIR does not support momentum.')
        super().__init__(y, operators, denoiser, config, weights_config, ref)

        self.lamb     = float(getattr(config, "diffpir_lambda",   7.0))
        self.sigma_n  = float(getattr(config, "diffpir_sigma_n",  0.05))
        self.xi       = float(getattr(config, "diffpir_xi",       0.0))   # DDIM stochasticity
        self.cg_iters = int(getattr(config,   "diffpir_cg_iters", 30))
        self.proximal = getattr(config, "diffpir_proximal", "gradient")   # 'gradient' or 'cg'

        self.sigma_next  = None
        self._init_xt    = True
        self.xt          = None
        self.x_im1       = None

    def set_next_sigma(self, sigma_next):
        if sigma_next is None:
            self.sigma_next = None
            return
        self.sigma_next = float(sigma_next) if not torch.is_tensor(sigma_next) else float(sigma_next)

    def _data_proximal(self, x0, sigma):
        """
        Proximal step for the data term:
          x* = argmin_x ||Ax - y||^2 + rho * ||x - x0||^2
          rho = 2 * lambda * sigma_n^2 / sigma^2
        """
        rho = 2.0 * self.lamb * (self.sigma_n ** 2) / (sigma ** 2)
        if self.proximal == 'gradient':
            # Single gradient descent step on the quadratic (matches InverseBench reference)
            res  = self.forward_op(x0) - self.y
            grad = self.adjoint_op(res)   # A^T(Ax0 - y)
            return x0 - grad / rho
        elif self.proximal == 'cg':
            # Exact solution via CG: (A^T A + rho I) x = A^T y + rho * x0
            rhs    = self.adjoint_op(self.y) + rho * x0
            normal = lambda x: self.normal_op(x) + rho * x
            cg_solve = ConjugateGradient(normal, self.cg_iters, dbprint=False)
            return cg_solve(x0.detach().clone(), rhs)
        else:
            raise ValueError(f'Unknown diffpir_proximal: {self.proximal}')

    def estimate_update(self, noise_std):
        self.iter += 1
        sigma     = float(noise_std)       # float for arithmetic
        sigma_ten = torch.as_tensor(noise_std).to(self.device)  # tensor for denoiser

        # Initialize xt ~ N(0, sigma_max^2 I) at the first step
        if self._init_xt or (self.xt is None):
            self.xt = torch.randn(self.nbatch, 2, self.nx, self.ny,
                                  device=self.device, dtype=self.x.dtype) * sigma
            self._init_xt = False

        sigma_next = float(self.sigma_next) if self.sigma_next is not None else 0.0

        with torch.no_grad():
            # 1) Prior denoising step
            x0 = self.denoiser(self.xt, sigma_ten, None)

            # 2) HQS data proximal
            x0hat = self._data_proximal(x0, sigma)

            # 3) DDIM update toward sigma_next
            if sigma_next > 0:
                effect = (self.xt - x0hat) / sigma   # predicted noise direction
                if self.xi > 0:
                    eps      = torch.randn_like(self.xt)
                    noise_dir = (self.xi ** 0.5) * eps + ((1.0 - self.xi) ** 0.5) * effect
                else:
                    noise_dir = effect
                self.x_im1 = x0hat + noise_dir * sigma_next
            else:
                # Last step: return clean estimate
                self.x_im1 = x0hat

            # Debug buffers
            self.e_obs     = self.forward_op(x0hat) - self.y
            self.e_denoise = self.xt - x0hat

        self.sigma_next = None

    def step(self):
        self.xt = self.x_im1
        self.x  = self.x_im1

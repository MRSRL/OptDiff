"""Natural-image baselines DDNM, DiffPIR, DPS, DDRM and RED-diff, adapted from the RED-diff code base.

Each sampler runs its whole reverse trajectory in sample(x, y, ts, y_0=...), where ts are the DDPM timesteps in
ascending order, and returns the lists (x_t iterates, x_0 predictions) with the reconstruction first.
"""
import numpy as np
import torch

from utils.degradations_natimg import Inpainting


class DiffusionModel:
    """Epsilon-prediction network with its DDPM schedule; returns the noise and the predicted x0.

    Unconditional case of RED-diff's ClassifierGuidanceModel: y (class label) and scale (classifier guidance
    scale) are unused.
    """

    def __init__(self, model, diffusion):
        self.model = model
        self.diffusion = diffusion

    def __call__(self, xt, y, t, scale=1.0):
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        et = self.model(xt, t)[:, :3]
        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        return et, x0_pred


class DDNM:
    """
    DDNM / DDNM+: Denoising Diffusion Null-Space Model.
    Paper: https://arxiv.org/abs/2212.00490

    Implements both Algorithm 1 (DDNM, noise-free) and Algorithm 2 (DDNM+, noisy)
    using the SVD-based degradation operators from this repository.

    DDNM refines only the null-space contents during the reverse diffusion process:
      x0_hat = A†y + (I - A†A)x0_pred                              (Eq. 13)

    DDNM+ extends DDNM with:
      - Scaled range-space correction for noisy measurements          (Eq. 17)
      - Per-singular-value noise balancing via Lambda/Lambda_noise    (Eq. 64-65)
      - Time-travel trick for improved restoration quality            (Sec. 3.3)

    When sigma_y=0 and travel_length=0, DDNM+ reduces exactly to DDNM.
    """

    def __init__(self, model, H, sigma_y, cfg):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.cfg = cfg
        self.eta = cfg.eta
        self.sigma_y = sigma_y
        self.travel_length = cfg.travel_length
        self.travel_repeat = cfg.travel_repeat

    def initialize(self, x, y, ts, **kwargs):
        return torch.randn_like(x)

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        n = x.size(0)
        H = self.H
        singulars = H.singulars()

        xt = self.initialize(x, y, ts)

        # Inpainting uses the scalar-lambda step (Sec. 3.3), which works in image space with H and H_pinv.
        # The other degradations use the per-singular-value step.
        use_scalar = isinstance(H, Inpainting)

        # ts is a sorted ascending list of actual diffusion timesteps, e.g. [0, 10, ..., 990].
        # The time-travel schedule works with indices into ts (0 = lowest noise, N-1 = highest).
        # Each index i maps to actual timestep ts[i]; index -1 maps to timestep -1 (final step).
        ts_list = list(ts)
        idx_pairs = self._get_index_schedule(len(ts_list), self.travel_length, self.travel_repeat)

        xt_s = [xt.cpu()]
        x0_s = []
        x0_preds_cache = []

        for i_idx, j_idx in idx_pairs:
            if j_idx < i_idx:
                # --- Normal reverse diffusion step ---
                ti = ts_list[i_idx]
                si = ts_list[j_idx] if j_idx >= 0 else -1

                t = torch.ones(n, dtype=torch.long, device=x.device) * ti
                s = torch.ones(n, dtype=torch.long, device=x.device) * si
                alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
                alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)

                # Eq. 12: predict x0 and noise from xt
                et, x0_pred = self.model(xt, y, t, scale=1.0)

                if self.sigma_y > 0:
                    # DDNM+: SVD-domain correction + noise balancing (Algo. 2),
                    # or the scalar-lambda step for inpainting.
                    if use_scalar:
                        xt, x0_hat = self._ddnm_plus_step_scalar(
                            H, x0_pred, y_0, et, alpha_s
                        )
                    else:
                        xt, x0_hat = self._ddnm_plus_step(
                            H, x0_pred, y_0, et, alpha_s, singulars
                        )
                else:
                    # DDNM: exact range-null space decomposition (Algo. 1)
                    x0_hat = self._ddnm_correction(H, x0_pred, y_0)
                    c1 = (1 - alpha_s).sqrt() * self.eta
                    c2 = (1 - alpha_s).sqrt() * np.sqrt(1 - self.eta ** 2)
                    xt = alpha_s.sqrt() * x0_hat + c1 * torch.randn_like(xt) + c2 * et

                x0_preds_cache.append(x0_pred)
                x0_s.append(x0_hat.cpu())
                xt_s.append(xt.cpu())
            else:
                # --- Time-travel: re-noise from last x0 prediction ---
                sj = ts_list[j_idx]
                s = torch.ones(n, dtype=torch.long, device=x.device) * sj
                alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
                x0_last = x0_preds_cache[-1]
                xt = alpha_s.sqrt() * x0_last + (1 - alpha_s).sqrt() * torch.randn_like(x0_last)
                xt_s.append(xt.cpu())

        return list(reversed(xt_s)), list(reversed(x0_s))

    # ------------------------------------------------------------------
    # Core correction methods
    # ------------------------------------------------------------------

    def _ddnm_correction(self, H, x0_pred, y_0):
        """DDNM (noise-free): Eq. 13.

        x0_hat = x0_pred - A†(A x0_pred - y)
               = A†y + (I - A†A) x0_pred
        """
        residual = H.H(x0_pred) - y_0.reshape(y_0.size(0), -1)
        correction = H.H_pinv(residual).view(*x0_pred.size())
        return x0_pred - correction

    def _ddnm_plus_step_scalar(self, H, x0_pred, y_0, et, alpha_s):
        """DDNM+ reverse step with scalar lambda (Sec. 3.3, Eq. 19).

        Used for inpainting. Operates in image space via H/H_pinv directly.

        Scalar lambda: lambda_t = min(1, sigma_s * sqrt(1 - eta²) / (a_s * sigma_y))
        x0_hat = x0_pred - lambda_t * A†(A x0_pred - y)
        """
        sigma_s = (1 - alpha_s).sqrt().view(-1)[0]
        a_s = alpha_s.sqrt().view(-1)[0]
        eta = self.eta
        sigma_y = self.sigma_y

        lambda_t = min(1.0, (sigma_s * np.sqrt(1 - eta ** 2) / (a_s * sigma_y + 1e-10)).item())

        residual = H.H(x0_pred) - y_0.reshape(y_0.size(0), -1)
        correction = lambda_t * H.H_pinv(residual).view(*x0_pred.size())
        x0_hat = x0_pred - correction

        c1 = sigma_s * eta
        c2 = sigma_s * np.sqrt(1 - eta ** 2)
        xt_next = a_s * x0_hat + c1 * torch.randn_like(x0_pred) + c2 * et

        return xt_next, x0_hat

    def _ddnm_plus_step(self, H, x0_pred, y_0, et, alpha_s, singulars):
        """DDNM+ reverse step with per-singular-value scaling (Appendix I, DDIM version).

        Computes:
          1. Lambda-scaled correction for x0_hat  (Eq. 17, 64)
          2. Properly balanced noise term          (Eq. 65)
          3. Combined step: xt_next = sqrt(alpha_s) * x0_hat + noise

        Returns (xt_next, x0_hat).
        """
        n = x0_pred.size(0)
        m = singulars.shape[0]
        sigma_y = self.sigma_y
        eta = self.eta

        sigma_s = (1 - alpha_s).sqrt().view(-1)[0]
        a_s = alpha_s.sqrt().view(-1)[0]

        # --- Transform to V-domain ---
        Vt_x0 = H.Vt(x0_pred)   # (batch, D)
        Ut_y  = H.Ut(y_0)        # (batch, m)
        Vt_et = H.Vt(et)         # (batch, D)
        D = Vt_x0.shape[1]

        # Singular value helpers
        nonzero = singulars > 1e-10
        inv_sing = torch.zeros_like(singulars)
        inv_sing[nonzero] = 1.0 / singulars[nonzero]

        # Per-SV threshold: sqrt(alpha_s) * sigma_y / s_i
        threshold = torch.full_like(singulars, float('inf'))
        threshold[nonzero] = a_s * sigma_y * inv_sing[nonzero]

        # Partition singular values (Eq. 64)
        large = nonzero & (sigma_s >= threshold)
        small = nonzero & (sigma_s < threshold)
        zero  = ~nonzero

        # Convert boolean masks (size m) to integer indices for indexing into
        # D-dimensional tensors. Boolean masks of size m cannot directly index
        # a dimension of size D (e.g. m=12288, D=196608 for SR4).
        large_idx = large.nonzero(as_tuple=False).squeeze(1)
        small_idx = small.nonzero(as_tuple=False).squeeze(1)
        zero_idx  = zero.nonzero(as_tuple=False).squeeze(1)

        # Lambda_t per singular value (Eq. 64)
        lambda_t = torch.ones_like(singulars)
        if small_idx.numel() > 0:
            lambda_t[small_idx] = (
                singulars[small_idx] * sigma_s * np.sqrt(1 - eta ** 2)
                / (a_s * sigma_y + 1e-10)
            )

        # --- x0_hat in V-domain (Eq. 17) ---
        spectral_residual = singulars.unsqueeze(0) * Vt_x0[:, :m] - Ut_y[:, :m]
        correction = lambda_t.unsqueeze(0) * inv_sing.unsqueeze(0) * spectral_residual

        Vt_x0_hat = Vt_x0.clone()
        Vt_x0_hat[:, :m] -= correction

        # --- Noise in V-domain (Eq. 65) ---
        eps = torch.randn(n, D, device=x0_pred.device)
        noise_Vt = torch.zeros(n, D, device=x0_pred.device)

        # Case 1: large nonzero SVs — full correction, reduced noise
        if large_idx.numel() > 0:
            gamma_sq = (
                sigma_s ** 2
                - alpha_s.view(-1)[0] * sigma_y ** 2
                  * lambda_t[large_idx] ** 2 / singulars[large_idx] ** 2
            )
            noise_Vt[:, large_idx] = gamma_sq.clamp(min=0).sqrt().unsqueeze(0) * eps[:, large_idx]

        # Case 2: small nonzero SVs — reduced correction, eta-scaled noise
        if small_idx.numel() > 0:
            noise_Vt[:, small_idx] = sigma_s * eta * eps[:, small_idx]

        # Case 3: zero SVs within [0, m) — standard DDIM noise
        if zero_idx.numel() > 0:
            noise_Vt[:, zero_idx] = (
                sigma_s * np.sqrt(1 - eta ** 2) * Vt_et[:, zero_idx]
                + sigma_s * eta * eps[:, zero_idx]
            )

        # Case 4: null-space components beyond m — standard DDIM noise
        if D > m:
            noise_Vt[:, m:] = (
                sigma_s * np.sqrt(1 - eta ** 2) * Vt_et[:, m:]
                + sigma_s * eta * eps[:, m:]
            )

        # --- Combine and transform back ---
        xt_next = H.V(a_s * Vt_x0_hat + noise_Vt).view(*x0_pred.size())
        x0_hat  = H.V(Vt_x0_hat).view(*x0_pred.size())

        return xt_next, x0_hat

    # ------------------------------------------------------------------
    # Time schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _get_index_schedule(N, travel_length, travel_repeat):
        """Build a schedule of index pairs over ts[0..N-1].

        Index N-1 = highest noise level (first denoising step).
        Index 0   = lowest noise level (last denoising step).
        Index -1  = sentinel for the final step (alpha = 1).

        Returns list of (from_idx, to_idx) pairs:
          from_idx > to_idx  →  normal reverse step
          from_idx < to_idx  →  time-travel (re-noise)
        """
        if travel_length <= 0 or travel_repeat <= 0:
            # Simple descending schedule: (N-1, N-2), ..., (1, 0), (0, -1)
            idxs = list(range(N - 1, -1, -1)) + [-1]
            return list(zip(idxs[:-1], idxs[1:]))

        jumps = {}
        for j in range(0, N - travel_length, travel_length):
            jumps[j] = travel_repeat - 1

        idx = N
        idxs = []
        while idx >= 1:
            idx -= 1
            idxs.append(idx)
            if jumps.get(idx, 0) > 0:
                jumps[idx] -= 1
                for _ in range(travel_length):
                    idx += 1
                    idxs.append(idx)

        idxs.append(-1)
        return list(zip(idxs[:-1], idxs[1:]))


class DiffPIR:
    """
    DiffPIR: Denoising Diffusion Models for Plug-and-Play Image Restoration
    Paper: https://arxiv.org/abs/2305.08995

    Integrates the HQS plug-and-play framework into DDIM-style reverse diffusion.
    At each step:
      1. Predict x0 from xt using the diffusion model.
      2. Solve the data proximal subproblem (HQS data term) to get x0hat.
      3. Compute corrected noise estimate and step to x_{t-1}.
    """

    def __init__(self, model, H, sigma_y, cfg):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.cfg = cfg
        self.sigma_y = sigma_y               # noise std of measurement (sigma_n in paper)
        self.lamb = cfg.lamb                 # lambda: regularization weight
        self.zeta = cfg.zeta                 # zeta: noise injection balance (xi in paper)
        self._svd_available = None           # delay _svd_available check to sample time

    @staticmethod
    def _check_svd(H):
        """Check whether the degradation model supports SVD-based exact proximal solve."""
        try:
            H.singulars()
            return True
        except NotImplementedError:
            return False

    def _solve_proximal_svd(self, H, y_0, x0_pred, rho_t):
        """Exact closed-form solution to the proximal subproblem using SVD.

        Solves:  argmin_x  ||y - H(x)||^2  +  rho_t ||x - x0_pred||^2
        Solution: x* = (H^T H + rho I)^{-1} (H^T y + rho x0_pred)
        In the SVD basis this reduces to element-wise division by (sigma_i^2 + rho).
        """
        s = H.singulars()
        n_batch = x0_pred.shape[0]
        rho = rho_t.view(n_batch, 1)          # (batch, 1)

        # 1-D singulars shared across the batch
        m = s.shape[0]
        s_row = s.unsqueeze(0)                # (1, m)

        Vt_x0 = H.Vt(x0_pred)                # (batch, data_dim)
        Ut_y  = H.Ut(y_0)                    # (batch, ≤ m)

        numerator   = s_row * Ut_y[:, :m] + rho * Vt_x0[:, :m]
        denominator = s_row ** 2 + rho

        Vt_result = Vt_x0.clone()
        Vt_result[:, :m] = numerator / denominator
        # Components beyond m stay as Vt_x0 (null-space → keep prior)

        return H.V(Vt_result).view(x0_pred.shape)

    def sample(self, x, y, ts, **kwargs):

        if self._svd_available is None:
            self._svd_available = self._check_svd(self.H)

        y_0 = kwargs["y_0"]
        n = x.size(0)
        H = self.H
        sigma_n = self.sigma_y

        xt = self.initialize(x, y, ts, y_0=y_0)

        ss = [-1] + list(ts[:-1])
        xt_s = [xt.cpu()]
        x0_s = []

        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n, dtype=torch.long, device=xt.device) * ti
            s = torch.ones(n, dtype=torch.long, device=xt.device) * si

            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)   # alpha_bar_t
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)   # alpha_bar_{t-1}
            sigma_bar_t_sq = (1 - alpha_t) / alpha_t
            rho_t = self.lamb * (sigma_n ** 2) / (sigma_bar_t_sq + 1e-8)  # add small epsilon for stability

            # Step 1: predict x0 from the diffusion model
            with torch.no_grad():
                et, x0_pred = self.model(xt, y, t, scale=1.0)

            # Step 2: data proximal subproblem
            #   argmin_x  ||y - H(x)||^2  +  rho_t ||x - x0_pred||^2
            if self._svd_available:
                # Exact closed-form solve via SVD decomposition
                x0hat = self._solve_proximal_svd(H, y_0, x0_pred.detach(), rho_t)
            else:
                # Gradient-based approximation for non-SVD degradations
                x0_pred_g = x0_pred.detach().requires_grad_(True)
                with torch.enable_grad():
                    residual = y_0 - H.H(x0_pred_g)
                    data_loss = (residual ** 2).sum()
                    grad = torch.autograd.grad(data_loss, x0_pred_g)[0]
                if rho_t.mean().item() > 1e-4:
                    x0hat = x0_pred.detach() - grad.detach() / (2 * rho_t)
                else:
                    x0hat = x0_pred.detach()

            # Step 3: DDIM step with corrected noise
            eps_hat = (xt - alpha_t.sqrt() * x0hat) / (1 - alpha_t).sqrt()
            eps_t = torch.randn_like(xt)
            xt_next = (alpha_s.sqrt() * x0hat
                       + (1 - alpha_s).sqrt() * (
                           np.sqrt(1 - self.zeta) * eps_hat
                           + np.sqrt(self.zeta) * eps_t
                       ))

            xt = xt_next
            xt_s.append(xt.detach().cpu())
            x0_s.append(x0hat.detach().cpu())

        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize(self, x, y, ts, **kwargs):
        # DiffPIR starts from pure Gaussian noise at the highest noise level
        n = x.size(0)
        ti = ts[-1]
        t = torch.ones(n, dtype=torch.long, device=x.device) * ti
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        return (1 - alpha_t).sqrt() * torch.randn_like(x)


class DPS:
    """
    DPS: Diffusion Posterior Sampling for General Noisy Inverse Problems
    Paper: https://arxiv.org/abs/2209.14687

    DDIM reverse steps from the network's x0 prediction, each followed by a gradient step on the measurement
    residual ||y - H(x0_pred(xt))|| with respect to xt, differentiated through the network.
    At each step:
      1. Predict x0 and the noise from xt.
      2. Take the DDIM step to x_{t-1} (eta sets its stochasticity).
      3. Subtract grad_term_weight * grad_xt ||y - H(x0_pred)||^2 / ||y - H(x0_pred)||.
    """

    def __init__(self, model, H, sigma_y, cfg):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.cfg = cfg                                  # sigma_y is not used by DPS
        self.grad_term_weight = cfg.grad_term_weight   # step size, divided by the residual norm
        self.eta = cfg.eta                             # DDIM stochasticity

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        n = x.size(0)
        H = self.H

        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        xt_s = [x.cpu()]
        x0_s = []

        xt = x
        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n, dtype=torch.long, device=x.device) * ti
            s = torch.ones(n, dtype=torch.long, device=x.device) * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            xt = xt.clone().requires_grad_(True)

            et, x0_pred = self.model(xt, y, t, scale=1.0)
            mat_norm = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()
            mat = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum()

            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]
            coeff = self.grad_term_weight / mat_norm.reshape(-1, 1, 1, 1)

            grad_term = grad_term.detach()
            xs = alpha_s.sqrt() * x0_pred.detach() + c1 * torch.randn_like(xt) + c2 * et.detach() - grad_term * coeff
            xt_s.append(xs.detach().cpu())
            x0_s.append(x0_pred.detach().cpu())
            xt = xs

        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize(self, x, y, ts, **kwargs):
        # DPS starts from the pseudo-inverse of the measurement, noised to the highest noise level
        y_0 = kwargs['y_0']
        n = x.size(0)
        x_0 = self.H.H_pinv(y_0).view(*x.size()).detach()
        ti = ts[-1]
        t = torch.ones(n, dtype=torch.long, device=x.device) * ti
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)


class DDRM:
    """
    DDRM: Denoising Diffusion Restoration Models
    Paper: https://arxiv.org/abs/2201.11793

    Works in the SVD of H: x is updated through its spectral components V^T x, and the measurement through U^T y,
    where component i of Sigma^-1 U^T y is component i of x plus noise of std sigma_y / s_i. At each step, with
    sigma_s the noise level (std of x_s / sqrt(alpha_s)) of the next step and x0 the network's prediction:
      - missing components (s_i = 0): DDIM step on x0 (eta sets its stochasticity);
      - components where y is less noisy than x_s (s_i * sigma_s > sigma_y): eta_b * (Sigma^-1 U^T y)_i
        + (1 - eta_b) * (V^T x0)_i, plus the noise missing to reach sigma_s;
      - components where y is noisier than x_s (s_i * sigma_s < sigma_y): (V^T x0)_i pulled toward y by
        (U^T y - Sigma V^T x0)_i / sigma_y, scaled by the deterministic part of the step.
    x_T is drawn from the same split, given y.

    sigma_y is the measurement noise std on the [-1, 1] image scale (data_config.input_noise), the noise main_natimg.py
    adds.
    """

    def __init__(self, model, H, sigma_y, cfg):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.sigma_y = sigma_y  # measurement noise std assumed by the updates
        self.eta = cfg.eta      # stochasticity of the steps on missing components and those where y is noisier than x_s
        self.eta_b = cfg.eta_b  # weight of the measurement on the components where y is less noisy than x_s, in [0, 1]

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        x = torch.randn_like(x)
        y_0 = kwargs["y_0"]
        sigma_y = self.sigma_y
        eta = self.eta
        eta_b = self.eta_b
        n = x.size(0)
        H = self.H

        # setup vectors used in the algorithm
        singulars = H.singulars().view(-1)
        Sigma = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=x.device)
        Sigma[: singulars.shape[0]] = singulars
        U_t_y = H.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]

        # initialize x_T as given in the paper
        T = torch.ones(n, dtype=torch.long, device=x.device) * ts[-1]
        largest_alphas = self.diffusion.alpha(T).view(-1, 1, 1, 1)
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_y)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_y / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init)
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], device=x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        # setup iteration variables
        x = H.V(init_y.view(x.size(0), -1)).view(*x.size())
        ss = [-1] + list(ts[:-1])
        x0_s = []
        xt_s = [x.cpu()]

        # iterate over the timesteps
        xt = x
        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n, dtype=torch.long, device=x.device) * ti
            s = torch.ones(n, dtype=torch.long, device=x.device) * si
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)

            et, x0_pred = self.model(xt, y, t)

            # variational inference conditioned on y
            sigma_s = (1 - alpha_s).sqrt()[0, 0, 0, 0] / alpha_s.sqrt()[0, 0, 0, 0]
            V_t_x0 = H.Vt(x0_pred)
            SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_s > sigma_y
            cond_after_lite = singulars * sigma_s < sigma_y
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_s * eta
            sigma_tilde_nextC = torch.sqrt(sigma_s ** 2 - std_nextC ** 2)

            std_nextA = sigma_s * eta
            sigma_tilde_nextA = torch.sqrt(sigma_s ** 2 - std_nextA ** 2)

            diff_sigma_t_nextB = torch.sqrt(sigma_s ** 2 - sigma_y ** 2 / singulars[cond_before_lite] ** 2 * (eta_b ** 2))

            # missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            # x_s less noisy than y
            Vt_xt_mod_next[:, cond_after] = (
                V_t_x0[:, cond_after]
                + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_y)[:, cond_after_lite]
                + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            )

            # x_s noisier than y
            Vt_xt_mod_next[:, cond_before] = (
                Sig_inv_U_t_y[:, cond_before_lite] * eta_b
                + (1 - eta_b) * V_t_x0[:, cond_before]
                + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite]
            )

            # aggregate all 3 cases and give next prediction
            xt_mod_next = H.V(Vt_xt_mod_next)
            xt = (alpha_s.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_s.append(x0_pred.cpu())
            xt_s.append(xt.cpu())

        return list(reversed(xt_s)), list(reversed(x0_s))


class REDDIFF:
    """
    RED-diff: A Variational Perspective on Solving Inverse Problems with Diffusion Models
    Paper: https://arxiv.org/abs/2305.04391

    Fits the mean mu of q(x0 | y) = N(mu, sigma_x0^2 I) with Adam, one step per timestep from the noisiest, starting
    from mu = H_pinv(y) (initial_estimate pinv, as in RED-diff) or Ht(y) (adjoint). At timestep t, with
    x0 = mu + sigma_x0 * noise and x_t = sqrt(alpha_t) x0 + sqrt(1 - alpha_t) eps:
      loss = obs_weight * mean((y - H(x0))^2) / 2 + grad_term_weight * w(t) * mean(stopgrad(eps_net(x_t, t) - eps) * x0),
    where w(t) is the function denoise_term_weight (see WEIGHTS) of sqrt(1 - alpha_t) / sqrt(alpha_t). The network is
    not differentiated. The reconstruction is the x0 of the last step, before its Adam step, as RED-diff's main.py
    reports it.
    """

    WEIGHTS = {'linear': lambda s: s, 'sqrt': torch.sqrt, 'square': torch.square, 'log': lambda s: torch.log(s + 1.0),
               'trunc_linear': lambda s: torch.clip(s, max=1.0), 'power2over3': lambda s: torch.pow(s, 2 / 3),
               'const': lambda s: torch.pow(s, 0.0)}

    def __init__(self, model, H, sigma_y, cfg):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H                                      # sigma_y is not used by RED-diff
        self.lr = cfg.lr                                # Adam step size
        self.grad_term_weight = cfg.grad_term_weight    # weight of the denoising term
        self.obs_weight = cfg.obs_weight                # weight of the measurement term
        self.sigma_x0 = cfg.sigma_x0                    # std of q(x0 | y)
        if cfg.initial_estimate not in ('pinv', 'adjoint'):
            raise ValueError(f'initial_estimate must be pinv or adjoint, not {cfg.initial_estimate}')
        self.initial_estimate = cfg.initial_estimate    # starting mu: H_pinv(y) or Ht(y)
        if cfg.denoise_term_weight not in self.WEIGHTS:
            raise ValueError(f'denoise_term_weight must be one of {list(self.WEIGHTS)}, not {cfg.denoise_term_weight}')
        self.weight_fn = self.WEIGHTS[cfg.denoise_term_weight]

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        n = x.size(0)
        H = self.H

        mu0 = H.H_pinv(y_0) if self.initial_estimate == 'pinv' else H.Ht(y_0)
        mu = mu0.view(*x.size()).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)

        for ti in reversed(ts):
            t = torch.ones(n, dtype=torch.long, device=x.device) * ti
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)

            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)
            x0_pred = mu + self.sigma_x0 * noise_x0
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt

            et, _ = self.model(xt, y, t)
            et = et.detach()

            e_obs = y_0 - H.H(x0_pred)
            loss_obs = (e_obs ** 2).mean() / 2
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()

            snr_inv = (1 - alpha_t[0]).sqrt() / alpha_t[0].sqrt()
            w_t = self.grad_term_weight * self.weight_fn(snr_inv)
            loss = w_t * loss_noise + self.obs_weight * loss_obs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return [x0_pred.detach()], [mu.detach()]


def build_baseline(name, model, H, sigma_y, cfg):
    if name == 'ddnm':
        return DDNM(model, H, sigma_y, cfg)
    elif name == 'diffpir':
        return DiffPIR(model, H, sigma_y, cfg)
    elif name == 'dps':
        return DPS(model, H, sigma_y, cfg)
    elif name == 'ddrm':
        return DDRM(model, H, sigma_y, cfg)
    elif name == 'reddiff':
        return REDDIFF(model, H, sigma_y, cfg)
    else:
        raise ValueError(f'No baseline named {name}')

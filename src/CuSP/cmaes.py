"""

Batch CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

Provides a BatchCMAES class with an interface similar to BatchLM,

designed for batch Stokes inversion and other optimization problems.

CMA-ES is a derivative-free, population-based evolutionary algorithm

that adapts a multivariate normal search distribution (mean and covariance)

over generations. Key advantages over LM:

  - No gradient/Jacobian computation needed

  - Robust to noisy objective functions

  - Good at escaping local minima (global search capability)

  - Only requires forward evaluations

Supports optional group-wise parameter normalization via config:

  config = {

      'decomposition': [Nt, Nt, Nt, Nt, Nt, 1, 1],  # group sizes summing to Nx

      'scalings': ['lin', 'lin', 'log', 'lin', 'lin', 'lin', 'lin'],  # 'lin' or 'log' per group

  }

Reference:

  Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.

  arXiv:1604.00772

"""

from dataclasses import dataclass

from typing import Callable, Optional, Dict, Any, List

import torch

import numpy as np

import time

import math
import numpy as np


Tensor = torch.Tensor

ObjectiveFn = Callable[[Tensor], Tensor]

@dataclass

class CMAESResult:

    """Result container matching LMResult pattern."""

    params: Tensor       # (Nb, Nx) best parameters found

    losses: Tensor       # (Nb,) final losses

    initial_losses: Tensor  # (Nb,) losses at initial guess

    converged: Tensor    # (Nb,) convergence flags

    sigma: Tensor        # (Nb,) final step sizes

    iterations: int      # total generations used

class BatchCMAES:

    """

    Batch Covariance Matrix Adaptation Evolution Strategy.

    Runs independent CMA-ES instances for each batch element.

    Uses separable (diagonal) covariance by default for memory efficiency

    in high-dimensional parameter spaces.

    CMA-ES algorithm follows Hansen (2016) with:

      - Weighted recombination

      - Cumulative Step-size Adaptation (CSA)

      - Rank-one + rank-mu covariance update

      - Separable (diagonal) covariance mode for Nx > 50

      - Optional group-wise parameter normalization via config

    """

    def __init__(

        self,

        max_iters: int = 500,

        pop_size: Optional[int] = None,

        init_sigma: float = 0.25,

        min_sigma: float = 1e-12,

        max_sigma: float = 1e12,

        tol_x: float = 1e-10,

        tol_fun: float = 1e-12,

        covariance_mode: str = "sep",

        patience: int = 50,

        forward: Optional[ObjectiveFn] = None,

        verbose: bool = True,

    ) -> None:

        if forward is None:

            raise ValueError("Please provide a forward model via `forward=`")

        self.max_iters = max_iters

        self._pop_size = pop_size

        self.init_sigma = init_sigma

        self.min_sigma = min_sigma

        self.max_sigma = max_sigma

        self.tol_x = tol_x

        self.tol_fun = tol_fun

        self.covariance_mode = covariance_mode.lower()

        if self.covariance_mode not in ("full", "sep"):

            raise ValueError("covariance_mode must be 'full' or 'sep'")

        self.patience = patience

        self.forward = forward

        self.verbose = verbose

        self.device = None

        self.dtype = None

        self.log: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    # Parameter normalization (group-wise via decomposition/scalings)

    # ------------------------------------------------------------------

    @staticmethod

    def _compute_normalization(

        x0: Tensor,

        decomposition: List[int],

        scalings: List[str],

        exploration_factor: float = 0.05,

        abs_floor: float = 1e-3,

        norm_std: Optional[List[Optional[float]]] = None,
        bounds: Optional[Dict[int, List[float]]] = None,


    ) -> Dict[str, Any]:

        """

        Compute per-group normalization statistics from initial guess.

        For each group a single scalar mean+std is computed (across all elements

        in the group over all batch samples). This ensures robustness regardless

        of batch size (Nb >= 1) and gives every parameter in a group the same

        normalization scale.

        Parameters

        ----------

        x0 : Tensor (Nb, Nx)

            Initial parameter guess used to estimate scale.

        decomposition : list of int

            Group sizes summing to Nx.

        scalings : list of str

            'lin' or 'log' per group (size-1 groups forced to 'lin').

        Returns

        -------

        norm_info : dict with keys:

            groups : list of slice objects

            transforms : list of str ('lin'|'log')

            mean : Tensor (n_groups,) scalar mean per group

            std  : Tensor (n_groups,) scalar std per group
            bounds_info : dict (optional)
                Mapping from group index to (lo_norm, hi_norm) for bounded groups.


        """

        cumsum = np.cumsum([0] + decomposition).tolist()

        n_groups = len(decomposition)

        norm_info = {

            'groups': [],

            'transforms': [],

            'mean': torch.zeros(n_groups, device=x0.device, dtype=x0.dtype),

            'std': torch.ones(n_groups, device=x0.device, dtype=x0.dtype),
'bounded': {}


        }

        for g in range(n_groups):

            sidx, eidx = cumsum[g], cumsum[g + 1]

            gsize = eidx - sidx

            transform = 'lin' if gsize == 1 else scalings[g]

            xg = x0[:, sidx:eidx]  # (Nb, gsize)

            if transform == 'log':

                xg_safe = xg.clamp(min=1e-30)

                xg_t = torch.log(xg_safe)

            else:

                xg_t = xg

            # Use scalar statistics (over all elements in the group) for robustness

            mean_g = xg_t.mean()

            # Nb * gsize samples -> use correction=0 (population std)

            std_g = xg_t.std(correction=0)

            # ---- Relative std floor for flat/small-variance groups ----

            # When a parameter (e.g. vLos) is constant across all layers,

            # std = 0, making CMA-ES unable to explore. Floor to

            # exploration_factor * |mean| for |mean| > 1e-6, else abs_floor.

            mean_abs = mean_g.abs().clamp(min=1e-12)

            # abs_floor can be float (global) or list (per-group)
            _af = abs_floor[g] if isinstance(abs_floor, list) else abs_floor
            std_floor = torch.where(

                mean_abs > 1e-6,

                exploration_factor * mean_abs,

                torch.full_like(mean_abs, _af)

            )

            std_g = torch.maximum(std_g, std_floor).clamp(min=1e-12)

            norm_info['groups'].append(slice(sidx, eidx))

            norm_info['transforms'].append(transform)

            norm_info['mean'][g] = mean_g

            # ---- norm_std override (user-specified scale) ----

            if norm_std is not None and norm_std[g] is not None:

                std_g = torch.tensor(norm_std[g], device=x0.device, dtype=x0.dtype)

            norm_info['std'][g] = std_g

            # ---- Per-group bounds transform ----
            if bounds is not None and g in bounds:
                lo_phys, hi_phys = bounds[g]
                if transform == 'log':
                    lo_t = max(math.log(max(lo_phys, 1e-300)), -700)
                    hi_t = min(math.log(max(hi_phys, 1e-300)), 700)
                else:
                    lo_t, hi_t = lo_phys, hi_phys
                lo_norm = (lo_t - mean_g.item()) / std_g.item()
                hi_norm = (hi_t - mean_g.item()) / std_g.item()
                norm_info['bounded'][g] = (lo_norm, hi_norm)


        return norm_info

    @staticmethod

    def _normalize_params(x: Tensor, norm_info: Dict[str, Any]) -> Tensor:

        """Transform physical-space parameters to normalized space ~O(1)."""

        x_norm = x.clone()

        for g, sl, trans in zip(

            range(len(norm_info['groups'])),

            norm_info['groups'], norm_info['transforms'],

        ):



            mn = norm_info['mean'][g]

            sd = norm_info['std'][g]

            xg = x[:, sl]

            if trans == 'log':

                xg_t = torch.log(xg.clamp(min=1e-30))

            else:

                xg_t = xg

            x_norm[:, sl] = (xg_t - mn) / sd

            # ---- Bounds transform: logit for bounded groups ----
            bounded = norm_info.get('bounded', {})
            if g in bounded:
                lo_norm, hi_norm = bounded[g]
                eps = 1e-12
                ratio = (x_norm[:, sl] - lo_norm) / max(hi_norm - lo_norm, 1e-300)
                ratio = torch.clamp(ratio, eps, 1.0 - eps)
                x_norm[:, sl] = torch.log(ratio / (1.0 - ratio))


        return x_norm

    @staticmethod

    def _unnormalize_params(x_norm: Tensor, norm_info: Dict[str, Any]) -> Tensor:

        """Transform normalized-space parameters back to physical space."""

        x = x_norm.clone()

        for g, sl, trans in zip(

            range(len(norm_info['groups'])),

            norm_info['groups'], norm_info['transforms'],

        ):


            # ---- Inverse bounds transform: sigmoid for bounded groups ----
            bounded = norm_info.get('bounded', {})
            if g in bounded:
                lo_norm, hi_norm = bounded[g]
                xg_n_sig = torch.sigmoid(x[:, sl])
                x[:, sl] = lo_norm + (hi_norm - lo_norm) * xg_n_sig
            mn = norm_info['mean'][g]

            sd = norm_info['std'][g]

            xg_n = x[:, sl]

            xg_t = xg_n * sd + mn

            if trans == 'log':

                xg = torch.exp(xg_t)

            else:

                xg = xg_t

            x[:, sl] = xg

        return x

    # ------------------------------------------------------------------

    # CMA-ES parameter helpers

    # ------------------------------------------------------------------

    @staticmethod

    def _default_pop_size(Nx: int) -> int:

        return max(5, 4 + int(3 * math.log(Nx)))

    @staticmethod

    def _cma_params(Nx: int, lam: int) -> Dict[str, float]:

        """Standard CMA-ES strategy parameters (Hansen 2016, Alg. 1)."""

        mu = lam // 2

        weights_raw = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]

        sum_w = sum(weights_raw)

        weights = [w / sum_w for w in weights_raw]

        mueff = 1.0 / sum(w * w for w in weights)

        cs = (mueff + 2.0) / (Nx + mueff + 5.0)

        ds = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (Nx + 1.0)) - 1.0) + cs

        cc = (4.0 + mueff / Nx) / (Nx + 4.0 + 2.0 * mueff / Nx)

        c1 = 2.0 / ((Nx + 1.3) ** 2 + mueff)

        cmu = min(1.0 - c1,

                  2.0 * (mueff - 2.0 + 1.0 / mueff) / ((Nx + 2.0) ** 2 + 2.0 * mueff / 2.0))

        chiN = math.sqrt(Nx) * (1.0 - 1.0 / (4.0 * Nx) + 1.0 / (21.0 * Nx * Nx))

        return dict(

            lam=lam, mu=mu,

            weights=weights, mueff=mueff,

            cs=cs, ds=ds, cc=cc, c1=c1, cmu=cmu, chiN=chiN,

        )

    # ------------------------------------------------------------------

    # Chi-squared computation

    # ------------------------------------------------------------------

    @staticmethod

    def _Chi2(Y: Tensor, y: Tensor, sig: Tensor) -> Tensor:

        return ((Y - y).square() / sig).sum(dim=1)

    @staticmethod

    def _check_losses(losses: Tensor, batch_size: int) -> Tensor:

        if losses.ndim == 2 and losses.shape[1] == 1:

            losses = losses[:, 0]

        elif losses.ndim == 1:

            pass

        else:

            raise ValueError(

                f"objective output must be shape (B,1) or (B,), got {tuple(losses.shape)}"

            )

        if losses.shape[0] != batch_size:

            raise ValueError(

                f"objective batch mismatch: expected {batch_size}, got {losses.shape[0]}"

            )

                # Replace NaN/Inf with a very large value (numerically safe)

        losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, 1e30))

        return losses

    # ------------------------------------------------------------------

    # Main optimization

    # ------------------------------------------------------------------

    def __call__(

        self,

        Y: Tensor,

        guess: Tensor,

        **kwargs,

    ) -> Tensor:

        """

        Run batched CMA-ES optimization.

        Parameters

        ----------

        Y : Tensor (Nb, Ny)

            Target observations (flattened Stokes).

        guess : Tensor (Nb, Nx)

            Initial parameter guess (mean of search distribution).

        **kwargs:

            config  : dict with optional keys:

                decomposition     : list of int  (group sizes summing to Nx)

                scalings          : list of str  ("lin"|"log" per group)

                exploration_factor : float (default 0.05) relative std floor

                                      for flat parameter groups.

                abs_floor         : float (default 1e-3) absolute std floor

                norm_std         : list of float|None per group (default None)
                init_std         : dict (optional)
                      Per-group normalization std override (direct specification).
                      Keys are group indices, values are std values.
                      Overrides the computed std for the group, directly controlling
                      the first-generation spread in physical space.
                      Example: {1: 0.1, 3: 0.5, 4: 0.5}
                      If both norm_std and init_std given, init_std overrides
                      specific groups.

                sigma_scales     : list of float (optional)
                      Per-group multiplier on init_sigma for exploration.
                      Default 1.0 for all groups.
                per_group_sigma  : dict (optional)
                      Per-group sigma values (direct specification, not multipliers).
                      Keys are group indices, values are sigma values in normalized space.
                      More intuitive than sigma_scales for setting exploration variance.
                      Example: {1: 0.2, 3: 0.1, 4: 0.1}
                      If both sigma_scales and per_group_sigma given,
                      per_group_sigma overrides specific groups.
                bounds           : dict (optional)
                      Per-group bounds in physical space.
                      Keys are group indices, values are [lo, hi].
                      Uses sigmoid/logit so CMA-ES stays unbounded.
                      Example: {3: [0.0, 3.14159], 4: [0.0, 3.14159]}
                first_gen_init   : dict (optional)
                      Per-group first-generation sampling override.
                      Keys are group indices, values are dicts with keys:
                        'std'    : float (required) sampling std in physical space
                        'mean'   : float or None (optional) center value for sampling;
                                   if None, uses the initial guess mean.
                        'bounds' : [lo, hi] (optional) clamp sampled values.
                      This overrides the first generation's population values for
                      the specified groups in PHYSICAL space, independent of the
                      normalization. Subsequent generations evolve normally via CMA-ES.
                      Example: {1: {'std': 0.2},
                                 3: {'std': 0.1, 'bounds': [0.0, 3.14159]},
                                 4: {'std': 0.1, 'bounds': [0.0, 3.14159]}}

            sig     : Tensor  noise variance for chi2

            isPrint : bool    print progress

            device  : torch.device

            dtype   : torch.dtype

        Returns

        -------

        x_best : Tensor (Nb, Nx)  best parameters found (physical space).

        """

        Nb, Ny = Y.shape

        Nx = guess.size(1)

        device = self.device or kwargs.get("device", Y.device)

        dtype = self.dtype or kwargs.get("dtype", Y.dtype)

        isPrint = kwargs.get("isPrint", self.verbose)

        sig = kwargs.get("sig", torch.tensor(1.0, device=device, dtype=dtype))

        config = kwargs.get("config", {})

        self.config = config

        if sig.dim() == 1:

            sig = sig.unsqueeze(1) if sig.shape[0] == Nb else sig

        # ---- Optional group-wise normalization ----

        decomposition = config.get("decomposition", None)

        norm_info = None
        first_gen_init = None
        bounds_input = None

        if decomposition is not None:

            dec_sum = sum(decomposition)

            if dec_sum != Nx:

                raise ValueError(f"decomposition sum {dec_sum} != Nx {Nx}")

            scalings = config.get("scalings", ['lin'] * len(decomposition))

            if len(scalings) != len(decomposition):

                raise ValueError(

                    f"len(scalings)={len(scalings)} != len(decomposition)={len(decomposition)}"

                )

            scalings = [

                'lin' if decomposition[g] == 1 else scalings[g]

                for g in range(len(decomposition))

            ]

            exploration_factor = config.get("exploration_factor", 0.05)

            abs_floor = config.get("abs_floor", 1e-3)

            norm_std = config.get("norm_std", None)
            sigma_scales = config.get("sigma_scales", None)
            # Convert per_group_abs_floor dict to per-group abs_floor list
            _abs_floor_raw = config.get("abs_floor", 1e-3)
            per_group_abs_floor = config.get("per_group_abs_floor", None)
            if per_group_abs_floor is not None:
                n_groups_af = len(decomposition)
                if isinstance(_abs_floor_raw, (int, float)):
                    abs_floor = [float(_abs_floor_raw)] * n_groups_af
                else:
                    abs_floor = list(_abs_floor_raw)
                for g_idx, g_af in per_group_abs_floor.items():
                    if 0 <= g_idx < n_groups_af:
                        abs_floor[g_idx] = float(g_af)
            else:
                abs_floor = _abs_floor_raw
            per_group_sigma = config.get("per_group_sigma", None)
            init_std = config.get("init_std", None)
            first_gen_init = config.get("first_gen_init", None)
            # Convert per_group_sigma dict to sigma_scales list
            if per_group_sigma is not None:
                n_groups2 = len(decomposition)
                if sigma_scales is None:
                    sigma_scales = [1.0] * n_groups2
                for g_idx, g_sigma in per_group_sigma.items():
                    if 0 <= g_idx < n_groups2:
                        sigma_scales[g_idx] = g_sigma / self.init_sigma
            bounds_input = config.get("bounds", None)
            # ---- Inject small noise for zero-variance groups ----
            add_noise_uniform = config.get("add_noise_uniform", True)
            if add_noise_uniform and decomposition is not None:
                cumsum_af = np.cumsum([0] + decomposition).tolist()
                n_groups_af2 = len(decomposition)
                for g_noise in range(n_groups_af2):
                    sidx_af, eidx_af = cumsum_af[g_noise], cumsum_af[g_noise + 1]
                    xg_af = guess[:, sidx_af:eidx_af]
                    xg_std = xg_af.std(correction=0)
                    # Get per-group abs_floor value
                    if isinstance(abs_floor, list):
                        af_val = abs_floor[g_noise]
                    else:
                        af_val = abs_floor
                    if xg_std < af_val:
                        noise = torch.randn_like(xg_af) * af_val
                        guess[:, sidx_af:eidx_af] = xg_af + noise
                        # Clamp within physical bounds if specified
                        if bounds_input is not None and g_noise in bounds_input:
                            blo, bhi = bounds_input[g_noise]
                            guess[:, sidx_af:eidx_af] = guess[:, sidx_af:eidx_af].clamp(blo, bhi)
                        if isPrint:
                            std_after = guess[:, sidx_af:eidx_af].std(correction=0)
                            print(f"    group {g_noise}: std={xg_std:.6e} < af={af_val}, injected noise, std_after={std_after:.6e}", flush=True)

            # Convert init_std dict to norm_std list
            if init_std is not None:
                n_groups3 = len(decomposition)
                if norm_std is None:
                    norm_std = [None] * n_groups3
                for g_idx, g_std in init_std.items():
                    if 0 <= g_idx < n_groups3:
                        norm_std[g_idx] = float(g_std)
            norm_info = self._compute_normalization(

                guess, decomposition, scalings,

                exploration_factor=exploration_factor,

                abs_floor=abs_floor,

                norm_std=norm_std,
                bounds=bounds_input,


            )

            if isPrint:

                print(f"  Norm: {len(decomposition)} groups, scalings={scalings}", flush=True)
                if bounds_input:
                    for bg, (blo, bhi) in bounds_input.items():
                        print(f"    group {bg}: bounds=[{blo:.4f}, {bhi:.4f}]", flush=True)
                if init_std:
                    for g, gs in init_std.items():
                        print(f"    group {g}: init_std={gs:.4f}", flush=True)
                if per_group_sigma:
                    for g, gs in per_group_sigma.items():
                        print(f"    group {g}: sigma={gs:.4f}", flush=True)
                elif sigma_scales:
                    print(f"    sigma_scales={sigma_scales}", flush=True)
                if first_gen_init:
                    for g, spec in first_gen_init.items():
                        g_std = spec.get("std", "?")
                        g_bnd = spec.get("bounds", None)
                        info = f"std={g_std}"
                        if g_bnd:
                            info += f" bounds=[{g_bnd[0]:.4f},{g_bnd[1]:.4f}]"
                        print(f"    group {g}: first_gen_init ({info})", flush=True)


        # ---- Strategy parameters ----

        lam = self._pop_size or self._default_pop_size(Nx)

        p = self._cma_params(Nx, lam)

        mu = p["mu"]

        weights = torch.tensor(p["weights"], device=device, dtype=dtype)

        mueff = p["mueff"]

        cs, ds = p["cs"], p["ds"]

        cc, c1, cmu = p["cc"], p["c1"], p["cmu"]

        chiN = p["chiN"]

        Y = Y.to(device=device, dtype=dtype)

        # ---- Normalize initial guess ----

        guess_d = guess.detach().clone().to(device=device, dtype=dtype)


        # ---- Build per-parameter sigma scale factor ----
        sigma_scale_param = torch.ones(Nx, device=device, dtype=dtype)
        if decomposition is not None and sigma_scales is not None:
            cumsum_ss = [0] + list(np.cumsum(decomposition))
            for g, ss in enumerate(sigma_scales):
                if ss is not None and ss != 1.0:
                    sidx, eidx = cumsum_ss[g], cumsum_ss[g+1]
                    sigma_scale_param[sidx:eidx] = ss

        if norm_info is not None:


            x_mean = self._normalize_params(guess_d, norm_info)

        else:

            x_mean = guess_d

        # ---- Initial evaluation (physical space) ----

        if norm_info is not None:

            x0_phys = self._unnormalize_params(x_mean, norm_info)

        else:

            x0_phys = x_mean

        with torch.no_grad():

            y0 = self.forward(x0_phys)

        sig_use = sig.to(device=device, dtype=dtype)

        if sig_use.dim() == 1:

            sig_use = sig_use.unsqueeze(1) if sig_use.shape[0] == Nb else sig_use

        init_chi2 = self._check_losses(self._Chi2(Y, y0, sig_use), Nb)

        # ---- State initialization (all in normalized space) ----

        C_diag = torch.ones(Nb, Nx, device=device, dtype=dtype)

        pc = torch.zeros(Nb, Nx, device=device, dtype=dtype)

        ps = torch.zeros(Nb, Nx, device=device, dtype=dtype)

        sigma = torch.full((Nb,), self.init_sigma, device=device, dtype=dtype)

        x_best = x_mean.clone()

        loss_best = init_chi2.clone()

        chi2_hist = [float(init_chi2.mean().item())]

        time0 = time.time()

        _sqrt_cc_term = math.sqrt(cc * (2.0 - cc) * mueff)

        _sqrt_cs_term = math.sqrt(cs * (2.0 - cs) * mueff)

        waiting = 0

        iterations = 0

        if isPrint:

            self._print_header(Nb, Nx, lam, init_chi2, time0, sigma, norm_info)

        error_occurred = False
        error_message = ""

        for gen in range(self.max_iters):

            iterations = gen + 1

            # ----- 1. Sample offspring (normalized space) -----

            z = torch.randn(Nb, lam, Nx, device=device, dtype=dtype)

            scales = torch.sqrt(C_diag)

            x_trial_norm = (

                x_mean.unsqueeze(1)

                + sigma.unsqueeze(-1).unsqueeze(-1) * scales.unsqueeze(1) * sigma_scale_param.unsqueeze(0).unsqueeze(0) * z

            )

            # ----- 2. Evaluate (physical space) -----

            x_flat_norm = x_trial_norm.reshape(-1, Nx)

            if norm_info is not None:

                x_flat_phys = self._unnormalize_params(x_flat_norm, norm_info)

            else:

                x_flat_phys = x_flat_norm

# ----- 1a. First-generation direct physical-space override -----
            if gen == 0 and first_gen_init is not None and norm_info is not None:
                # Direct physical-space sampling: bypass normalization constraints.
                # For specified groups, overwrite x_flat_phys with direct samples,
                # then convert back to normalized space so CMA-ES state stays consistent.
                for g_fgi, spec in first_gen_init.items():
                    sl = norm_info["groups"][g_fgi]
                    gsize = sl.stop - sl.start
                    std_phys = spec.get("std", 0.1)
                    # Use user-specified mean or compute from initial guess
                    if "mean" in spec:
                        mean_phys = spec["mean"]
                    else:
                        guess_phys_all = self._unnormalize_params(x_mean, norm_info)
                        mean_phys = guess_phys_all[:, sl].mean().item()
                    # Sample directly in physical space: phys = mean + std * N(0,1)
                    phys_samples = torch.randn(Nb, lam, gsize, device=device, dtype=dtype) * std_phys + mean_phys
                    # Apply bounds if specified
                    if "bounds" in spec:
                        lo_b, hi_b = spec["bounds"]
                        phys_samples = torch.clamp(phys_samples, lo_b, hi_b)
                    # Overwrite in x_flat_phys
                    x_flat_phys_rs = x_flat_phys.reshape(Nb, lam, Nx)
                    x_flat_phys_rs[:, :, sl] = phys_samples
                    x_flat_phys = x_flat_phys_rs.reshape(-1, Nx)
                # Convert modified physical values back to normalized space
                x_flat_norm = self._normalize_params(x_flat_phys, norm_info)
                # Update x_trial_norm to reflect the modified samples so that
                # sorting, recombination, and x_best tracking use the correct
                # normalized-space parameters.
                x_trial_norm = x_flat_norm.reshape(Nb, lam, Nx)
                # Recompute z from the modified x_trial_norm for consistency
                # with CSA / covariance updates.  z is defined as
                #   x_trial = x_mean + sigma * sqrt(C_diag) * sigma_scale * z
                # so z = (x_trial - x_mean) / (sigma * sqrt(C_diag) * sigma_scale)
                _scale = sigma.unsqueeze(-1).unsqueeze(-1) * scales.unsqueeze(1) * sigma_scale_param.unsqueeze(0).unsqueeze(0)
                _scale = _scale.clamp(min=1e-30)
                z = (x_trial_norm - x_mean.unsqueeze(1)) / _scale
            with torch.no_grad():

                y_flat = None
                try:
                    y_flat = self.forward(x_flat_phys)
                    if y_flat.shape != (Nb * lam, Ny):
                        y_flat = y_flat.reshape(Nb * lam, -1)
                        if y_flat.shape[1] != Ny:
                            y_flat = y_flat[:, :Ny]
                except Exception:
                    pass
                if y_flat is None:
                    y_flat = torch.full((Nb * lam, Ny), 1e10, device=x_flat_phys.device, dtype=x_flat_phys.dtype)
                    try:
                        for _i in range(lam):
                            x_group = x_flat_phys[_i::lam]
                            y_group = self.forward(x_group)
                            if y_group.dim() == 1:
                                y_group = y_group.unsqueeze(0)
                            y_flat[_i::lam] = y_group
                    except Exception as _e:
                        error_occurred = True
                        error_message = str(_e)
                        if isPrint:
                            print(f"  WARNING: forward evaluation failed at generation {iterations}, sample {_i}/{lam}: {type(_e).__name__}: {_e}", flush=True)
                            print(f"  Filling remaining samples with large chi2 and preparing for early stop.", flush=True)
                        # y_flat already initialized to 1e10; leave remaining as-is
                # Safety check
                if not torch.isfinite(y_flat).all():
                    y_flat = torch.where(torch.isfinite(y_flat), y_flat, torch.full_like(y_flat, 1e10))

            if y_flat.ndim == 2 and y_flat.shape[1] == Ny:

                pass

            elif y_flat.ndim == 2 and y_flat.shape[0] == Nb * lam and y_flat.shape[1] != Ny:

                y_flat = y_flat.reshape(Nb * lam, -1)

                if y_flat.shape[1] != Ny:

                    y_flat = y_flat[:, :Ny]

            Y_exp = Y.unsqueeze(1).expand(-1, lam, -1).reshape(-1, Ny).to(device=device, dtype=dtype)

            if sig_use.dim() == 2 and sig_use.shape[1] > 1:

                sig_exp = sig_use.unsqueeze(1).expand(-1, lam, -1).reshape(-1, Ny)

            else:

                sig_exp = sig_use

            y_flat = y_flat.reshape(-1, Ny)

            chi2_flat = self._check_losses(self._Chi2(Y_exp, y_flat, sig_exp), Nb * lam)

            chi2_pop = chi2_flat.reshape(Nb, lam)

            # ----- 3. Sort by fitness -----

            sorted_vals, sorted_idx = torch.sort(chi2_pop, dim=1)

            # ----- 4. Weighted recombination -----

            best_idx = sorted_idx[:, :mu]

            x_best_mu = x_trial_norm.gather(1, best_idx[:, :, None].expand(-1, -1, Nx))

            x_mean_new = (x_best_mu * weights[None, :, None]).sum(dim=1)

            # ----- 5. Update best-ever -----

            loss_gen = sorted_vals[:, 0]

            best_x_gen = x_trial_norm.gather(

                1, best_idx[:, 0:1, None].expand(-1, -1, Nx)

            ).squeeze(1)

            improved = loss_gen < loss_best

            x_best = torch.where(improved.unsqueeze(-1), best_x_gen, x_best)

            loss_best_old = loss_best
            loss_best = torch.where(improved, loss_gen, loss_best)

            # ----- 6. Step-size control (CSA) -----

            z_best_mu = z.gather(1, best_idx[:, :, None].expand(-1, -1, Nx))

            dz = (z_best_mu * weights[None, :, None]).sum(dim=1)

            ps = (1.0 - cs) * ps + _sqrt_cs_term * dz

            ps_norms = torch.linalg.norm(ps, dim=1)

            sigma = sigma * torch.exp(cs / ds * (ps_norms / chiN - 1.0))

            sigma = torch.clamp(sigma, self.min_sigma, self.max_sigma)

            # ----- 7. Covariance adaptation (corrected: proper decay + weighted sum) -----

            pc = (1.0 - cc) * pc + _sqrt_cc_term * dz

            if cmu > 0:

                z_best_w_sq = (z_best_mu.square() * weights[None, :, None]).sum(dim=1)

                C_diag = ((1.0 - c1 - cmu) * C_diag

                          + cmu * C_diag * z_best_w_sq

                          + c1 * pc * pc)

            else:

                C_diag = (1.0 - c1) * C_diag + c1 * pc * pc

            C_diag = C_diag.clamp(min=1e-16)

            # ----- 8. Update mean -----
            # CMA-ES mean is updated every generation toward the weighted
            # recombination of the best mu offspring. This is the standard
            # behavior; restricting updates to "improving" generations would
            # freeze the search and prevent exploration.
            x_mean = x_mean_new

            # ----- 9. Convergence check -----
            # Convergence is triggered when the best-found loss stops improving
            # for `patience` generations (i.e. loss_best plateaus).
            loss_new = sorted_vals[:, 0]

            chi2_hist.append(loss_new.mean().item())

            improved_any = improved.any()
            if not improved_any:
                waiting += 1
                if waiting >= self.patience:
                    if isPrint:
                        print(f"  >>> Converged after {iterations} generations (patience={self.patience})", flush=True)
                    break
            else:
                waiting = 0

            # conv: per-batch flag indicating if this batch has stopped improving
            conv = ~improved

            if isPrint and (gen % 25 == 0 or gen < 5):

                mi = float(loss_new.mean().item())

                ms = float(sigma.mean().item())

                nc = int(conv.sum().item())

                et = time.time() - time0

                print(f"  {iterations:>4d} | {mi:>16.6e} | {ms:>12.6e} | {et:>8.1f} | {nc}/{Nb}", flush=True)

            # Early stop if an error occurred during evaluation
            if error_occurred:
                if isPrint:
                    print(f"  >>> Early stop at generation {iterations} due to evaluation error", flush=True)
                    print(f"  Error: {error_message}", flush=True)
                    print(f"  Returning best parameters found so far.", flush=True)
                break

        # ---- Final evaluation (physical space) ----

        if norm_info is not None:

            x_best_phys = self._unnormalize_params(x_best, norm_info)

        else:

            x_best_phys = x_best

        final_chi2 = loss_best.clone()
        try:
            with torch.no_grad():
                y_final = self.forward(x_best_phys)
            final_chi2 = self._check_losses(self._Chi2(Y, y_final, sig_use), Nb)
        except Exception as _e:
            if isPrint:
                print(f"  WARNING: final evaluation failed, using tracked best loss: {type(_e).__name__}: {_e}", flush=True)
            final_chi2 = loss_best.clone()

        self.log = {

            "chi2_history": chi2_hist,

            "generations": iterations,

            "elapsed": time.time() - time0,

            "x_best": x_best_phys,

            "loss_best": loss_best,

            "sigma_final": sigma,

            "norm_info": norm_info,

        }

        if isPrint:

            elapsed = time.time() - time0

            mi = float(init_chi2.mean().item())

            mf = float(final_chi2.mean().item())

            rd = (mi - mf) / mi * 100 if mi > 0 else 0

            print(f"  Final: {mi:.6e} -> {mf:.6e} ({rd:.2f}%)  [{elapsed:.1f}s]", flush=True)

        return x_best_phys

    def compute(self, Y, guess, **kwargs):

        return self.__call__(Y, guess, **kwargs)

    def optimize(self, Y, guess, **kwargs):

        return self.__call__(Y, guess, **kwargs)

    def _print_header(self, Nb, Nx, lam, init_chi2, time0, sigma, norm_info=None):

        has_norm = norm_info is not None

        print("=" * 80, flush=True)

        print(f"  BatchCMAES: Nb={Nb} Nx={Nx} pop_size={lam} "

              f"mode={self.covariance_mode}"

              f"{' normalized' if has_norm else ''}", flush=True)

        print(f"  max_iters={self.max_iters} init_sigma={self.init_sigma:.4e}", flush=True)

        print("=" * 80, flush=True)

        print(f"  {'iter':>4s} | {'chi2':>16s} | {'sigma':>12s} | {'time[s]':>8s} | {'conv':>6s}", flush=True)

        mi = float(init_chi2.mean().item())

        ms = float(sigma.mean().item())

        print(f"  {0:>4d} | {mi:>16.6e} | {ms:>12.6e} | {0:>8.1f} | {'---':>6s}", flush=True)
def make_batch_forward(rf_func, Wt_1d, ltt_1d, pt, **rf_kwargs):
    def _forward(x_batch):
        Nb_pt = pt.size(0)
        Nx_b = x_batch.size(0)
        if Nx_b != Nb_pt and Nx_b % Nb_pt == 0:
            n_rep = Nx_b // Nb_pt
            pt_exp = pt.repeat_interleave(n_rep, dim=0)
        else:
            pt_exp = pt
        y = rf_func(Wt_1d, ltt_1d, x_batch, pt_exp, **rf_kwargs)
        return y.reshape(y.size(0), -1)
    return _forward

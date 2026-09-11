from dataclasses import dataclass
from typing import Callable

import torch
import time
import numpy as np

Tensor = torch.Tensor
ObjectiveFn = Callable[[Tensor], Tensor]


@dataclass
class LMResult:
    params: Tensor
    losses: Tensor
    initial_losses: Tensor
    converged: Tensor
    damping: Tensor
    iterations: int


class BatchLM:
    """Batch Levenberg-Marquardt optimizer with improved damping strategies.

    Key improvements for marquardt damping mode:
    1. Gain-ratio-based damping: uses rho = actual/predicted reduction
       to guide damping updates (Nielsen/Mor approach).
       rho > 0.75: decrease damping; rho < 0.25: increase damping.
       This prevents damping from ratcheting up and causing stagnation.

    2. Robust per-group marquardt damping: H += lambda * |diag(H)| / max(|diag|)
       Uses max (not median) for more stable per-group normalization.

    3. Per-group gradient preconditioning: normalizes gradient across
       parameter groups (T, v, B, angles, mac, fill) for balanced steps.

    4. Relative gradient convergence: ||grad|| / ||grad_initial|| < tol
       for scale-invariant convergence detection.

    5. Backtracking line search with min-scale fallback for failed batches.

    6. Smarter damping reset: geometric mean of min and init damping.
    """

    def __init__(
        self,
        max_iters: int = 20,
        init_damping: float = 1e-6,
        damping_factor_increase: float = 3.0,
        damping_factor_decrease: float = 0.3,
        min_damping: float = 1e-12,
        max_damping: float = 1e12,
        eps_svd: float = 1e-6,
        tol_grad: float = 1e-8,
        tol_step: float = 1e-8,
        tol_loss: float = 1e-12,
        step_solver: str = "svd",
        damping_mode: str = "marquardt",
        damping_adapt: str = "standard",
        trust_region: bool = False,
        patience: int = 10,
        step_scale_min: float = 0.01,
        step_scale_max: float = 1.0,
        step_scale_factor: float = 0.5,
        max_line_search: int = 6,
        adaptive_init_damping: bool = True,
        grad_precond: bool = False,
        gain_ratio_threshold_lo: float = 0.25,
        gain_ratio_threshold_hi: float = 0.75,
        armijo_c: float = 1e-4,
        forward: ObjectiveFn = None,
    ) -> None:
        if forward is None:
            raise ValueError("Please give a callable synthesis method")
        damping_adapt = damping_adapt.lower()
        if damping_adapt not in {"standard", "delayed", "hybrid"}:
            raise ValueError("damping_adapt must be 'standard', 'delayed', or 'hybrid'")
        if step_solver not in {"svd", "solve"}:
            raise ValueError("step_solver must be 'svd' or 'solve'")
        if damping_mode not in {"marquardt", "identity"}:
            raise ValueError("damping_mode must be 'marquardt' or 'identity'")

        self.max_iters = max_iters
        self.init_damping = init_damping
        self.damping_factor_increase = damping_factor_increase
        self.damping_factor_decrease = damping_factor_decrease
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.tol_grad = tol_grad
        self.tol_step = tol_step
        self.tol_loss = tol_loss
        self.step_solver = step_solver.lower()
        self.patience = patience
        self.damping_mode = damping_mode.lower()
        self.damping_adapt = damping_adapt
        self.trust_region = trust_region
        self.step_scale_min = step_scale_min
        self.step_scale_max = step_scale_max
        self.step_scale_factor = step_scale_factor
        self.max_line_search = max_line_search
        self.adaptive_init_damping = adaptive_init_damping
        self.grad_precond = grad_precond
        self.gain_ratio_threshold_lo = gain_ratio_threshold_lo
        self.gain_ratio_threshold_hi = gain_ratio_threshold_hi
        self.armijo_c = armijo_c
        self.eps_svd = eps_svd
        self.forward = forward

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
        return losses

    @staticmethod
    def _Chi2(Y, y, sig):
        return ((Y - y).square() / sig).sum(dim=1)

    @staticmethod
    def _Jacobian(y, x, zoom_factor):
        Nb, Ny = y.shape
        Nx = x.size(1)
        jacobian = torch.zeros((Nb, Ny, Nx), device=x.device, dtype=x.dtype)
        for iy in range(Ny):
            jacobian[:, iy, :] = torch.autograd.grad(
                outputs=y[:, iy] * zoom_factor,
                inputs=x,
                grad_outputs=torch.ones_like(y[:, iy]),
                create_graph=False,
                retain_graph=True,
            )[0].detach() / zoom_factor
        return jacobian

    def _LMcoef(self, Y, x, **kwargs):
        sig = kwargs.get("sig", torch.tensor(1)).to(device=x.device, dtype=x.dtype)
        zoom_factor = kwargs.get("zoom_factor", 1e-15)
        x_req = x.clone().requires_grad_(True)
        y = self.forward(x_req)
        Nb, Ny = y.shape
        Nx = x.size(1)
        DOF = Ny - Nx
        if DOF <= 0:
            raise ValueError(f"Inversion DOF <= 0: {DOF}")
        chi2 = self._Chi2(Y, y, sig)
        chi2 = self._check_losses(chi2, Nb)
        # ---- Smooth barrier penalty for physical bounds ----
        bounds_penalty = self._compute_bounds_penalty(x_req)
        chi2 = chi2 + bounds_penalty
        # ------------------------------------------------------
        grad = torch.autograd.grad(
            outputs=chi2, inputs=x_req,
            grad_outputs=torch.ones_like(chi2),
            create_graph=False, retain_graph=True,
        )[0]
        Jacobian = self._Jacobian(y, x_req, zoom_factor)
        Hessian = torch.einsum("bni,bnj->bij", Jacobian / sig.unsqueeze(-1), Jacobian)
        return chi2.detach(), grad.detach(), Jacobian.detach(), Hessian.detach()

    def _compute_adaptive_init_damping(self, Hessian: Tensor) -> Tensor:
        diag = torch.diagonal(Hessian, dim1=1, dim2=2)
        diag_median = torch.median(diag.abs(), dim=1).values
        init_damp = diag_median * self.init_damping
        return torch.clamp(init_damp, min=self.min_damping, max=self.max_damping)

    def _apply_damping(self, Hessian: Tensor, damping: Tensor) -> Tensor:
        """Apply damping to Hessian matrix.

        marquardt mode: H_damped = H + lambda * |diag(H)| / median(|diag(H)|)_per_group
        Uses median (not max) for more representative group-scale normalization,
        and blends with identity to ensure every parameter gets minimum damping.
        identity mode: H_damped = H + lambda * I
        """
        Nx = Hessian.size(1)
        eye = torch.eye(Nx, device=Hessian.device, dtype=Hessian.dtype)[None, ...]
        if self.damping_mode == "marquardt":
            diag = torch.diagonal(Hessian, dim1=1, dim2=2)
            decomposition = self.config.get("decomposition", [Nx])
            marq_blend = self.config.get("marquardt_blend", 0.6)
            idx_dec = [0] + np.cumsum(decomposition).astype(int).tolist() + [Nx]
            damp_diag = torch.zeros_like(diag)
            for g in range(len(decomposition)):
                sidx, eidx = idx_dec[g], idx_dec[g + 1]
                g_diag = diag[:, sidx:eidx]
                # Use median for representative group scale (more robust than max)
                g_median = g_diag.abs().median(dim=1).values.clamp(min=1e-30)
                marq_part = g_diag.abs() / g_median[:, None]
                # Blend marquardt with identity: ensures every parameter gets
                # at least (1-marq_blend)*damping, preventing severe under-damping
                damp_diag[:, sidx:eidx] = damping[:, None] * (
                    marq_blend * marq_part + (1 - marq_blend))
            damp_mat = damp_diag.diag_embed()
            H = Hessian + damp_mat
        else:
            H = Hessian + damping[:, None, None] * eye
        return H

    def _adjust_damping(self, damping, improved, accepted, chi2_current, chi2_next,
                        chi2_rel_change, iteration, Y, x0, **kwargs):
        """Adjust damping using gain ratio (Nielsen/Morsapproach).

        Computes rho = actual_reduction / predicted_reduction and uses it to
        guide damping updates:
        - rho > 0.75: excellent step -> decrease damping
        - 0.25 <= rho <= 0.75: moderate step -> keep damping
        - rho < 0.25: poor step -> increase damping
        - rho <= 0: step worsened chi2 -> aggressive increase

        This replaces the simple improved/not-improved logic with a more
        nuanced gain-ratio approach that prevents the damping from growing
        too quickly and causing stagnation - the primary issue with the
        original marquardt mode.
        """
        # Compute gain ratio (actual / predicted reduction)
        # Use quadratic model: chi2(x0+dx) ~ chi2(x0) + g^T*dx + 0.5*dx^T*H*dx
        # Predicted reduction = -g^T*dx - 0.5*dx^T*H*dx
        sig = kwargs.get("sig", torch.tensor(1)).to(device=damping.device, dtype=damping.dtype)
        ared = chi2_current - chi2_next  # actual reduction
        grad = self.log["grad"].detach()
        dx = self.log["step"].detach()
        J = self.log["jacobian"].detach()
        dy = (J @ dx.unsqueeze(-1)).squeeze(-1)
        # pred = -grad^T * dx - sum((J @ dx)^2 / sig)  (correct quadratic model)
        linear_term = -torch.sum(grad * dx, dim=1)  # should be positive for descent
        quad_term = torch.sum(dy.square() / sig, dim=1)
                # Quadratic model for chi2 (no 1/2 factor):
        #   chi2(x+dx) ~ chi2(x) + g^T*dx + quad_term
        # So predicted reduction = -g^T*dx - quad_term  (MINUS)
        pred = linear_term - quad_term
                # pred < 0 means model predicts worsening -> rho = 0
        # pred > 0: compute gain ratio normally
        rho = torch.where(pred > 0, ared / pred.clamp(min=1e-15), torch.zeros_like(ared))
        rho = torch.nan_to_num(rho, nan=0, posinf=1.0, neginf=0.0)
        self.log["gain_ratio"] = rho

        # Trust-region mode (unchanged behavior)
        if self.trust_region:
            scaling = 1 - (2 * rho - 1) ** 3
            scaling = torch.where(scaling < 1 / 3, 1 / 3, scaling)
            scaling = torch.where(rho < 0, self.damping_factor_increase, scaling)
            scaling = scaling.clamp(min=self.damping_factor_decrease, max=self.damping_factor_increase)
            return (damping * scaling).clamp(min=self.min_damping, max=self.max_damping)

        # Gain-ratio-based adaptation (the key improvement)
        multiplier = torch.ones_like(damping)

        # rho > hi_threshold: excellent improvement -> decrease damping
        multiplier = torch.where(rho > self.gain_ratio_threshold_hi,
                                 self.damping_factor_decrease, multiplier)
        # rho < lo_threshold: poor improvement -> increase damping
        multiplier = torch.where(rho < self.gain_ratio_threshold_lo,
                                 self.damping_factor_increase, multiplier)
        # rho <= 0 or inf/nan: step was bad -> aggressive increase
        multiplier = torch.where((rho <= 0) | ~torch.isfinite(rho),
                                 self.damping_factor_increase * self.damping_factor_increase,
                                 multiplier)

        # Apply damping_adapt as secondary control
        if self.damping_adapt == "delayed":
            multiplier = torch.where(multiplier > 1, torch.ones_like(multiplier), multiplier)
        elif self.damping_adapt == "hybrid":
            if iteration <= self.max_iters // 3:
                multiplier = torch.where(multiplier > 1, torch.ones_like(multiplier), multiplier)

        new_damping = (damping * multiplier).clamp(min=self.min_damping, max=self.max_damping)

        # Stuck detection: track non-accepted streaks at ANY damping level.
        # Uses accepted (not improved) so non-monotonic near-best
        # steps that were accepted do not count as stuck.
        sc = self._stuck_count
        sc[~accepted] += 1
        sc[accepted] = 0
        # Reset if stuck for too long (at any damping)
        reset_stuck_thresh = self.config.get("stuck_reset_thresh", 8)
        reset_mask = sc >= reset_stuck_thresh
        if reset_mask.any():
            # Reset damping adaptively: geometric mean of current and init
            reset_val = torch.sqrt(new_damping[reset_mask].clamp(min=1e-15) * self.init_damping).clamp(
                min=self.min_damping * 10, max=self.init_damping * 100)
            new_damping[reset_mask] = reset_val
            sc[reset_mask] = 0

        # Damping recovery: detect when damping is at minimum and progress
        # is negligible. Uses chi2 change relative to BEST (not current)
        # for more robust detection of genuine plateaus.
        recovery_streak = self.config.get("damp_recovery_streak", 5)
        rel_change_tol = self.config.get("damp_recovery_rel_tol", 1e-6)
        near_min = damping <= self.min_damping * 100
        # Use change vs best chi2 (from caller scope via self.log) for robust plateau detection
        best_chi2 = self.log.get("_best_chi2", chi2_current)
        rel_change_vs_best = torch.abs(best_chi2 - chi2_next) / (torch.abs(best_chi2) + 1e-10)
        slow_progress = rel_change_vs_best < rel_change_tol
        sc_recovery = getattr(self, "_recovery_count", None)
        if sc_recovery is None:
            sc_recovery = torch.zeros_like(damping, dtype=torch.int)
            self._recovery_count = sc_recovery
        sc_recovery[accepted & near_min & slow_progress] += 1
        sc_recovery[~(accepted & near_min & slow_progress)] = 0
        recovery_mask = sc_recovery >= recovery_streak
        if recovery_mask.any():
            # Adaptive reset: geometric mean, scaled by chi2 ratio to be more aggressive
            # when far from best
            chi2_ratio = (chi2_next[recovery_mask] / best_chi2[recovery_mask].clamp(min=1e-10)).clamp(1.0, 100.0)
            reset_val = (damping[recovery_mask] * self.init_damping * chi2_ratio).sqrt().clamp(
                min=self.min_damping * 100, max=self.init_damping * 100)
            new_damping[recovery_mask] = reset_val
            sc[recovery_mask] = 0
            sc_recovery[recovery_mask] = 0
            if self.log.get("damp_recovery", None) is None:
                self.log["damp_recovery"] = []
            self.log["damp_recovery"].append(iteration)

        return new_damping

    def _line_search(self, Y, x0, delta, chi2_current, **kwargs):
        """Backtracking line search. Returns (delta, success_mask).

        Tries progressively smaller step scales. If no scale produces
        a valid improvement, returns zero delta for that batch.
        The success_mask indicates which batches found a valid step.
        """
        sig = kwargs.get("sig", torch.tensor(1)).to(device=x0.device, dtype=x0.dtype)
        scale = torch.full((x0.size(0),), self.step_scale_max, device=x0.device, dtype=x0.dtype)
        found_valid = torch.zeros(x0.size(0), device=x0.device, dtype=torch.bool)

        # Try full step
        try:
            x_full = x0 + delta
            chi2_full = self._Chi2(Y, self.forward(x_full), sig).detach()
            valid = (chi2_full < chi2_current) & torch.isfinite(chi2_full)
            found_valid = valid
            need_search = ~valid
        except Exception:
            need_search = torch.ones(x0.size(0), device=x0.device, dtype=torch.bool)

        for _ in range(self.max_line_search):
            if not need_search.any():
                break
            scale[need_search] *= self.step_scale_factor
            scale.clamp_(min=self.step_scale_min)
            try:
                x_try = x0 + delta * scale[:, None]
                chi2_try = self._Chi2(Y, self.forward(x_try), sig).detach()
                valid = (chi2_try < chi2_current) & torch.isfinite(chi2_try)
            except Exception:
                valid = torch.zeros(x0.size(0), device=x0.device, dtype=torch.bool)
            found_valid = found_valid | valid
            need_search = need_search & ~valid

        # For batches that found a valid step, use their scale
        # For batches that never found a valid step, use gradient descent 
        # fallback with step_scale_min (zero delta causes stagnation)
        zero_delta = torch.zeros_like(delta)
        gd_fallback = delta / (delta.norm(dim=1, keepdim=True).clamp(min=1e-15)) * self.step_scale_min
        step = torch.where(found_valid[:, None], delta * scale[:, None], gd_fallback)
        self.log["line_search_success"] = found_valid
        return step

    def _Stepping(self, Y, x, **kwargs):
        """Compute the LM step with optional gradient preconditioning."""
        x0 = x.detach().clone()
        chi2, grad, Jacobian, Hessian = self._LMcoef(Y, x0, **kwargs)
        self.log["chi2_current"] = chi2.detach()
        self.log["grad"] = grad.detach()
        self.log["hessian"] = Hessian
        self.log["jacobian"] = Jacobian

        # Gradient preconditioning: balance parameter groups
        # When parameters span different scales (T, v, B, angles, mac, fill),
        # raw gradient can be dominated by one group. Normalizing each group
        # to unit norm allows balanced contribution to the step.
        if self.grad_precond:
            decomposition = self.config.get("decomposition", [grad.size(1)])
            idx_dec = [0] + np.cumsum(decomposition).astype(int).tolist() + [grad.size(1)]
            grad_scaled = grad.clone()
            for g in range(len(decomposition)):
                sidx, eidx = idx_dec[g], idx_dec[g + 1]
                g_grad = grad[:, sidx:eidx]
                g_norm = g_grad.norm(dim=1, keepdim=True).clamp(min=self.tol_grad)
                grad_scaled[:, sidx:eidx] = g_grad / g_norm
            # Blend 50-50: gentle preconditioning
            grad = 0.5 * grad + 0.5 * grad_scaled

        # Zero gradient for frozen parameters so they do not affect step computation
        if getattr(self, "_frozen", False):
            grad = grad * (~self._freeze_mask).float()

        # ---- Bayesian regularization (Gaussian prior) ----
        # Pulls specified parameters toward target values with given strength.
        # Penalty: R(x) = 0.5 * strength * (x - target)^2
        #   dR/dx = strength * (x - target)
        #   d2R/dx2 = strength
        # This prevents turbulence parameters from being hijacked to
        # compensate for residual errors in atmospheric profiles.
        reg_indices = self.config.get("reg_indices", [])
        if reg_indices:
            reg_targets = self.config.get("reg_targets", [0.0] * len(reg_indices))
            reg_strengths = self.config.get("reg_strengths", [0.1] * len(reg_indices))
            for k, idx in enumerate(reg_indices):
                s = float(reg_strengths[k])
                t = float(reg_targets[k])
                # Clipped gradient contribution (pulls toward target)
                # Clipping prevents regularization from dominating data gradient
                # when parameters are far from target (e.g. micro at 50 vs target 0.8)
                dev = x0[:, idx] - t
                max_dev = max(abs(t) * 10.0, 1.0)  # cap at 10x target or 1.0
                dev = torch.clamp(dev, -max_dev, max_dev)
                grad[:, idx] = grad[:, idx] + s * dev
                # NOTE: Hessian diagonal is NOT modified.
                # Gradient-only regularization with clipping acts as a "soft prior"
                # that guides without dominating. 
            self.log["reg_applied"] = True
        # ------------------------------------------------

        # ---- Gradient boosting for weak-sensitivity groups ----
        boost_indices = self.config.get("grad_boost_indices", [])
        boost_factors = self.config.get("grad_boost_factors", [])
        if boost_indices and len(boost_factors) == len(boost_indices):
            for k, idx in enumerate(boost_indices):
                factor = float(boost_factors[k])
                if factor != 1.0:
                    grad[:, idx] = grad[:, idx] * factor
            self.log["boost_applied"] = True
        # -------------------------------------------------------

        if torch.isnan(Hessian).any():
            self.log.setdefault("check", {}).update(dict(
                nan=True,
                idx_nan=torch.where(torch.isnan(Hessian).any(dim=(1, 2)))[0],
            ))
            return x0.detach()
        self.log.setdefault("check", {}).update({"nan": False})
        Hessian = torch.nan_to_num(Hessian)
        grad = torch.nan_to_num(grad)

        H = self._apply_damping(Hessian, self.damping)
        if self.step_solver == "svd":
            delta = self._SVD(grad, H)
        else:
            try:
                delta = torch.linalg.solve(H, -grad.unsqueeze(-1)).squeeze(-1)
            except Exception:
                delta = torch.linalg.lstsq(H, -grad.unsqueeze(-1)).solution.squeeze(-1)
        delta = delta.detach()

        if self.max_line_search > 0:
            delta = self._line_search(Y, x0, delta, chi2, **kwargs)

        # Zero delta for frozen parameters
        if getattr(self, "_frozen", False):
            delta = delta * (~self._freeze_mask).float()

        x1 = (x0 + delta).detach()
        self.log["step"] = delta.detach()
        return x1


    def _SVD(self, grad, H):
        L, S, Rt = torch.linalg.svd(H, full_matrices=False)
        G = torch.diag_embed(S)
        decomposition = self.config["decomposition"]
        tol_singular = self.config["tol_singular"]
        Np = len(decomposition)
        if Np == 1:
            Smax = S.max(axis=1, keepdim=True)[0]
            G_inv = torch.diag_embed(
                torch.where(S <= tol_singular * Smax, 0, 1 / S)
            )
        else:
            Nx = grad.size(1)
            idx_dec = [0] + np.cumsum(decomposition).astype(int).tolist() + [Nx]
            G_dec = []
            for i in range(Np):
                sidx = idx_dec[i]
                eidx = idx_dec[i + 1]
                mask = torch.zeros_like(H, dtype=torch.bool)
                mask[..., sidx:eidx] = 1
                L_dec = L * mask
                Gi = G @ L_dec @ Rt
                G_diag = torch.diagonal(Gi, dim1=1, dim2=2)
                G_diag = torch.where(
                    G_diag <= tol_singular * G_diag.max(axis=1, keepdim=True)[0],
                    0, G_diag,
                )
                G_dec.append(G_diag)
            G_dec = torch.stack(G_dec, dim=0)
            gk = G_dec.sum(dim=0)
            gk_inv = torch.where(torch.isclose(gk, torch.zeros_like(gk)), 0, 1 / gk)
            G_inv = torch.diag_embed(gk_inv)
        H_inv = L @ G_inv @ Rt
        delta = torch.matmul(H_inv, -grad.unsqueeze(-1)).squeeze(-1)
        return delta.detach()

    def _apply_param_bounds(self, x):
        """Soft projection toward physical bounds (iterative).

        Unlike hard clamp, this applies a partial push-back iteratively
        until parameters converge inside the valid range. Parameters that
        drift out of bounds are gently nudged back, but are never forced
        exactly to the boundary value. Combined with the barrier penalty
        in _compute_bounds_penalty, this prevents parameters from getting
        stuck at boundaries while still respecting physical limits.

        Controlled by config keys:
          bounds_stiffness: fraction of violation corrected per call (default 0.7)
          bounds_iterations: max iterations for iterative correction (default 20)
          bounds_tolerance: stop when max violation < tolerance (default 1e-4)
          vlos_lower: lower bound for vLos group (default -30.0, i.e. -30 km/s)
          vlos_upper: upper bound for vLos group (default 30.0, i.e. 30 km/s)
          bounds_soft_margin: relative margin for soft boundaries (default 0.005)
        """
        decomposition = self.config.get("decomposition", [x.size(1)])
        cumsum = [0] + list(np.cumsum(decomposition))
        n_groups = len(decomposition)
        stiffness = self.config.get("bounds_stiffness", 0.7)
        soft_margin_ratio = self.config.get("bounds_soft_margin", 0.005)
        max_iters = self.config.get("bounds_iterations", 20)
        tol = self.config.get("bounds_tolerance", 1e-4)

        with torch.no_grad():
            for _iter in range(max_iters):
                max_violation = 0.0

                # --- vLos (group 1): [vlos_lower, vlos_upper] (default -30 to 30 km/s) ---
                if n_groups >= 2:
                    v_lower = self.config.get("vlos_lower", -30.0)
                    v_upper = self.config.get("vlos_upper", 30.0)
                    v_range = abs(v_upper - v_lower)
                    v_margin = max(v_range * soft_margin_ratio, 1e-3)
                    v_start, v_end = cumsum[1], cumsum[2]
                    v = x[:, v_start:v_end]
                    excess_high = torch.relu(v - (v_upper - v_margin))
                    excess_low = torch.relu((v_lower + v_margin) - v)
                    if excess_high.numel() > 0:
                        max_violation = max(max_violation, float(excess_high.max()))
                        max_violation = max(max_violation, float(excess_low.max()))
                    x[:, v_start:v_end] = v - stiffness * excess_high + stiffness * excess_low

                # --- gamma (group 3): [0, pi] ---
                if n_groups >= 4:
                    g_start, g_end = cumsum[3], cumsum[4]
                    margin_g = np.pi * soft_margin_ratio
                    g = x[:, g_start:g_end]
                    excess_high = torch.relu(g - (np.pi - margin_g))
                    excess_low = torch.relu(margin_g - g)
                    if excess_high.numel() > 0:
                        max_violation = max(max_violation, float(excess_high.max()))
                        max_violation = max(max_violation, float(excess_low.max()))
                    x[:, g_start:g_end] = g - stiffness * excess_high + stiffness * excess_low

                # --- phi (group 4): [0, pi] ---
                if n_groups >= 5:
                    p_start, p_end = cumsum[4], cumsum[5]
                    margin_p = np.pi * soft_margin_ratio
                    p = x[:, p_start:p_end]
                    excess_high = torch.relu(p - (np.pi - margin_p))
                    excess_low = torch.relu(margin_p - p)
                    if excess_high.numel() > 0:
                        max_violation = max(max_violation, float(excess_high.max()))
                        max_violation = max(max_violation, float(excess_low.max()))
                    x[:, p_start:p_end] = p - stiffness * excess_high + stiffness * excess_low

                # --- xi / microturbulence (group 5): >= 0 ---
                if n_groups >= 6:
                    xi_idx = cumsum[5]
                    margin_xi = soft_margin_ratio
                    xi = x[:, xi_idx:xi_idx+1]
                    excess_low = torch.relu(margin_xi - xi)
                    if excess_low.numel() > 0:
                        max_violation = max(max_violation, float(excess_low.max()))
                    x[:, xi_idx:xi_idx+1] = xi + stiffness * excess_low

                # --- zt / macro (group 6): >= 0 ---
                if n_groups >= 7:
                    zt_idx = cumsum[6]
                    margin_zt = soft_margin_ratio
                    zt = x[:, zt_idx:zt_idx+1]
                    excess_low = torch.relu(margin_zt - zt)
                    if excess_low.numel() > 0:
                        max_violation = max(max_violation, float(excess_low.max()))
                    x[:, zt_idx:zt_idx+1] = zt + stiffness * excess_low

                if max_violation < tol:
                    break

    def _compute_bounds_penalty(self, x):
        """Smooth barrier penalty for physical parameter bounds.

        Adds quadratic penalty when parameters approach or exceed bounds.
        This is a soft constraint that works alongside _apply_param_bounds
        to keep parameters within physical ranges without hard clamping.

        Controlled by config keys:
          bounds_penalty_strength: scalar penalty weight (default 1000.0)
          bounds_margin: soft margin before penalty activates (default 0.02)
          vlos_lower, vlos_upper: bounds for vLos (default -30, 30)
          vlos_penalty_strength: separate penalty strength for vLos
        """
        decomposition = self.config.get("decomposition", [x.size(1)])
        cumsum = [0] + list(np.cumsum(decomposition))
        n_groups = len(decomposition)
        strength = self.config.get("bounds_penalty_strength", 5000.0)
        margin = self.config.get("bounds_margin", 0.02)

        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        penalties = torch.zeros(batch_size, device=device, dtype=dtype)

        # --- vLos (group 1): [-30, 30] km/s penalty ---
        if n_groups >= 2:
            v_start, v_end = cumsum[1], cumsum[2]
            v = x[:, v_start:v_end]
            v_lower = self.config.get("vlos_lower", -30.0)
            v_upper = self.config.get("vlos_upper", 30.0)
            v_strength = self.config.get("vlos_penalty_strength", strength * 2)
            v_range = abs(v_upper - v_lower)
            v_margin = max(v_range * 0.005, 0.1)
            pen_low = torch.relu((v_lower + v_margin) - v) ** 2
            pen_high = torch.relu(v - (v_upper - v_margin)) ** 2
            penalties = penalties + v_strength * (pen_low.sum(dim=1) + pen_high.sum(dim=1))

        if n_groups >= 4:
            g_start, g_end = cumsum[3], cumsum[4]
            g = x[:, g_start:g_end]
            pen_low = torch.relu(margin - g) ** 2
            pen_high = torch.relu(g - (np.pi - margin)) ** 2
            penalties = penalties + strength * (pen_low.sum(dim=1) + pen_high.sum(dim=1))

        if n_groups >= 5:
            p_start, p_end = cumsum[4], cumsum[5]
            p = x[:, p_start:p_end]
            pen_low = torch.relu(margin - p) ** 2
            pen_high = torch.relu(p - (np.pi - margin)) ** 2
            penalties = penalties + strength * (pen_low.sum(dim=1) + pen_high.sum(dim=1))

        if n_groups >= 6:
            xi_idx = cumsum[5]
            xi = x[:, xi_idx:xi_idx+1]
            pen_low = torch.relu(margin - xi) ** 2
            penalties = penalties + strength * pen_low.sum(dim=1)

        if n_groups >= 7:
            zt_idx = cumsum[6]
            zt = x[:, zt_idx:zt_idx+1]
            pen_low = torch.relu(margin - zt) ** 2
            penalties = penalties + strength * pen_low.sum(dim=1)

        return penalties
    def __call__(self, Y, guess, **kwargs):
        """Run batched Levenberg-Marquardt optimization.

        All batches are optimized simultaneously with per-batch damping
        and convergence tracking.
        """
        Nb, Ny = Y.shape
        Nx = guess.size(1)
        config = kwargs.get("config", {})
        setattr(self, "config", dict(decomposition=[Nx], tol_singular=1.e-6))
        if kwargs.get("usrLMcoef", None) is not None:
            self._LMcoef = kwargs.get("usrLMcoef")
        self.config.update(config)
        # ---- Auto-populate reg_targets from initial guess ----
        if self.config.get("reg_from_initial", False):
            reg_idx = self.config.get("reg_indices", [])
            if reg_idx:
                targets = [guess[0, idx].item() for idx in reg_idx]
                self.config["reg_targets"] = targets
                if kwargs.get("isPrint", True):
                    print(f"  [REG] Auto-targets from initial guess: indices={reg_idx} targets={targets}", flush=True)
        # --------------------------------------------------------
        setattr(self, "device", kwargs.get("device", Y.device))
        setattr(self, "dtype", kwargs.get("dtype", Y.dtype))
        setattr(self, "damping", torch.full((Nb,), self.init_damping, device=self.device, dtype=self.dtype))
        setattr(self, "log", {})
        isPrint = kwargs.get("isPrint", True)
        x0 = guess.detach().clone().to(self.device).requires_grad_(False)
        # Apply physical bounds to initial guess (gamma/phi in [0,pi], xi/zt >= 0)
        self._apply_param_bounds(x0)
        Y = Y.to(self.device)
        converged = torch.zeros(Nb, device=self.device, dtype=torch.bool)
        sig = kwargs.get("sig", torch.tensor(1)).to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            init_chi2 = self._check_losses(self._Chi2(Y, self.forward(x0), sig), Nb)
        self.log["chi2_history"] = [init_chi2.mean().item()]

        iterations = 0
        time0 = time.time()
        damping = self.damping

        if self.adaptive_init_damping:
            _, _, _, H_init = self._LMcoef(Y, x0, **kwargs)
            with torch.no_grad():
                adaptive_damp = self._compute_adaptive_init_damping(H_init)
                damping = torch.where(
                    adaptive_damp > self.init_damping,
                    adaptive_damp,
                    torch.full_like(damping, self.init_damping),
                )
                self.damping = damping
            x0 = x0.detach().clone().requires_grad_(False)

        self._stuck_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
        self._recovery_count = torch.zeros(Nb, device=self.device, dtype=torch.int)

        # Track initial gradient for relative convergence
        _, init_grad, _, _ = self._LMcoef(Y, x0, **kwargs)
        init_grad_norm = torch.linalg.norm(init_grad, dim=1).clamp(min=self.tol_grad)

        if isPrint:
            self._print_header(Nb, Ny, Nx, iterations, init_chi2, time0, damping)

        waiting = 0
        # Track best chi2 per batch for non-monotonic acceptance
        best_chi2 = init_chi2.clone()
        best_x = x0.clone()
        # Counter for consecutive global non-improvement (used for perturbation)
        global_stall = 0
        chi2_tol_nonmono = self.config.get("nonmonotonic_tol", 0.01)

        # ---- Parameter freezing setup ----
        Nx_full = x0.size(1)

        # ---- Check for sequential unfreeze (phase-based) ----
        seq_unfreeze = self.config.get("sequential_unfreeze", False)
        if seq_unfreeze:
            # Use decomposition to determine group boundaries
            decomp = self.config.get("decomposition", [Nx_full])
            cumsum_decomp = [0] + list(np.cumsum(decomp))
            n_groups = len(decomp)

            # Phase tracking
            self._seq_phase = 0
            self._seq_n_groups = n_groups
            self._seq_cumsum = cumsum_decomp
            self._seq_phase_iters = 0
            self._seq_phase_best_chi2 = float("inf")
            self._seq_phase_stall = 0
            self._seq_phase_boundaries = [0]
            self._seq_phase_chi2 = []

            # Phase transition criteria
            self._seq_patience = self.config.get("seq_patience", 10)
            self._seq_max_iters = self.config.get("seq_max_iters_per_phase", 60)
            self._seq_chi2_tol = self.config.get("seq_chi2_tol", 1e-5)
            self._seq_min_iters = self.config.get("seq_min_iters_per_phase", 8)

            # Set initial freeze mask: freeze all groups except group 0
            freeze_mask = torch.ones(Nx_full, device=self.device, dtype=torch.bool)
            freeze_mask[cumsum_decomp[0]:cumsum_decomp[1]] = False
            self._freeze_mask = freeze_mask
            self._frozen = True

            # Compute group names for logging
            _default_names = ["T","vLOS","B","gamma","phi","xi","zeta"]
            group_names = []
            for g in range(n_groups):
                if g < len(_default_names):
                    group_names.append(_default_names[g])
                else:
                    group_names.append(f"grp{g}")

            if isPrint:
                free_names = [group_names[g] for g in range(1, n_groups) if cumsum_decomp[g+1] > cumsum_decomp[g]]
                msg = f"  [SEQ-UNFREEZE] Phase 0/{n_groups-1}: free={group_names[0]}, frozen={free_names}"
                print(msg, flush=True)

            # Auto-increase max_iters to accommodate all sequential unfreeze phases.
            # Each phase may consume up to seq_max_iters_per_phase iterations.
            # Without this, the outer loop (range(self.max_iters)) would terminate
            # before later phases ever start.
            min_iters_needed = n_groups * self._seq_max_iters
            if self.max_iters < min_iters_needed:
                old_max = self.max_iters
                self.max_iters = min_iters_needed
                if isPrint:
                    print(f"  [SEQ-UNFREEZE] max_iters auto-increased: {old_max} -> {self.max_iters} "
                          f"(need {n_groups} phases x {self._seq_max_iters} max iters/phase)", flush=True)

        # ---- Legacy single-stage freeze (backward compatible) ----
        elif self.config.get("freeze_indices", None) is not None:
            freeze_indices = self.config.get("freeze_indices", [])
            unfreeze_iter = self.config.get("unfreeze_iter", 0)
            freeze_mask = torch.zeros(Nx_full, device=self.device, dtype=torch.bool)
            if freeze_indices and unfreeze_iter > 0:
                freeze_mask[freeze_indices] = True
                self._freeze_mask = freeze_mask
                self._unfreeze_iter = unfreeze_iter
                self._frozen = True
                if isPrint:
                    msg = f"  [FREEZE] Locking params at indices {freeze_indices} for {unfreeze_iter} iterations"
                    print(msg, flush=True)
            else:
                self._frozen = False
        else:
            self._frozen = False
        # ------------------------------------------
        for i in range(self.max_iters):
            iterations += 1

            # ---- Sequential unfreeze: check phase transition ----
            if seq_unfreeze and self._frozen:
                chi2_cur_mean = (init_chi2.mean().item() if iterations == 1
                                 else self.log["chi2_current"].mean().item())
                do_transition = False

                # Track phase best chi2
                if chi2_cur_mean < self._seq_phase_best_chi2:
                    self._seq_phase_best_chi2 = chi2_cur_mean
                    self._seq_phase_stall = 0
                else:
                    self._seq_phase_stall += 1

                self._seq_phase_iters += 1

                # Transition criteria:
                if self._seq_phase_stall >= self._seq_patience and self._seq_phase_iters >= self._seq_min_iters:
                    do_transition = True
                if self._seq_phase_iters >= self._seq_max_iters:
                    do_transition = True
                if waiting >= self._seq_patience // 2:
                    do_transition = True

                if do_transition:
                    self._seq_phase += 1
                    self._seq_phase_iters = 0
                    self._seq_phase_stall = 0
                    self._seq_phase_best_chi2 = float("inf")
                    self._seq_phase_boundaries.append(iterations)
                    self._seq_phase_chi2.append(chi2_cur_mean)

                    if self._seq_phase >= self._seq_n_groups:
                        self._frozen = False
                        self._freeze_mask[:] = False
                        if isPrint:
                            msg = f"  [SEQ-UNFREEZE] Phase {self._seq_phase}/{n_groups-1}: ALL FREE (full opt)"
                            print(msg, flush=True)
                        reset_val = torch.full((Nb,), self.init_damping * 5, device=self.device, dtype=self.dtype)
                        damping = torch.clamp(reset_val, min=self.min_damping, max=self.max_damping)
                        self.damping = damping
                        self._stuck_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                        self._recovery_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                        waiting = 0
                        global_stall = 0
                        _, init_grad_full, _, _ = self._LMcoef(Y, x0, **kwargs)
                        init_grad_norm = torch.linalg.norm(init_grad_full, dim=1).clamp(min=self.tol_grad)
                        best_chi2 = self._check_losses(self._Chi2(Y, self.forward(x0), sig), Nb)
                    else:
                        cs = self._seq_cumsum
                        self._freeze_mask[cs[self._seq_phase]:cs[self._seq_phase+1]] = False
                        if isPrint:
                            free_groups = [group_names[g] for g in range(self._seq_phase+1)]
                            frozen_groups = [group_names[g] for g in range(self._seq_phase+1, n_groups)]
                            msg = f"  [SEQ-UNFREEZE] Phase {self._seq_phase}/{n_groups-1}: free={free_groups}, frozen={frozen_groups}"
                            print(msg, flush=True)
                        reset_val = torch.full((Nb,), self.init_damping * 5, device=self.device, dtype=self.dtype)
                        damping = torch.clamp(reset_val, min=self.min_damping, max=self.max_damping)
                        self.damping = damping
                        self._stuck_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                        self._recovery_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                        waiting = 0
                        global_stall = 0
                        _, init_grad_full, _, _ = self._LMcoef(Y, x0, **kwargs)
                        init_grad_norm = torch.linalg.norm(init_grad_full, dim=1).clamp(min=self.tol_grad)
                        best_chi2 = self._check_losses(self._Chi2(Y, self.forward(x0), sig), Nb)
            # ---- End sequential unfreeze ----
            x1 = self._Stepping(Y, x0, **kwargs)
            chi2_current = self.log["chi2_current"]
            chi2_next = self._Chi2(Y, self.forward(x1), sig).detach()
            improved = (chi2_next < chi2_current) & torch.isfinite(chi2_next)

            # Non-monotonic acceptance: also accept steps that are within tol of best chi2
            # This prevents stagnation when all steps are marginally worse
            near_best = (chi2_next < best_chi2 * (1 + chi2_tol_nonmono)) & torch.isfinite(chi2_next)
            accepted = improved | near_best

            # ---- Legacy unfreeze when unfreeze_iter is reached ----
            if (not seq_unfreeze) and self._frozen and hasattr(self, "_unfreeze_iter") and iterations == self._unfreeze_iter:
                self._frozen = False
                self._freeze_mask[:] = False
                if isPrint:
                    msg = f"  [UNFREEZE] Releasing frozen params at iter {iterations}, resetting damping"
                    print(msg, flush=True)
                reset_val = torch.full((Nb,), self.init_damping * 5, device=self.device, dtype=self.dtype)
                damping = torch.clamp(reset_val, min=self.min_damping, max=self.max_damping)
                self.damping = damping
                self._stuck_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                self._recovery_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                waiting = 0
                global_stall = 0
                _, init_grad_full, _, _ = self._LMcoef(Y, x0, **kwargs)
                init_grad_norm = torch.linalg.norm(init_grad_full, dim=1).clamp(min=self.tol_grad)
                best_chi2 = init_chi2.clone()
            # ----------------------------------------------

                        # Compute relative chi2 change for damping recovery detection
            rel_chi2_change = torch.abs(chi2_current - chi2_next) / (torch.abs(chi2_current) + 1e-10)
            damping = self._adjust_damping(
                damping, improved, accepted, chi2_current, chi2_next,
                rel_chi2_change, iterations, Y, x0, **kwargs,
            )
            self.damping = damping

            # Update parameters for accepted batches
            x1_detached = x1.detach().clone().requires_grad_(False)
            x0 = torch.where(accepted[:, None], x1_detached, x0)
            self._apply_param_bounds(x0)
            self._apply_param_bounds(x1_detached)
            best_x = torch.where((chi2_next < best_chi2)[:, None], x1_detached, best_x)
            best_chi2 = torch.where(chi2_next < best_chi2, chi2_next, best_chi2)
            self.log["_best_chi2"] = best_chi2

            # Convergence checks
            step_norm = torch.linalg.norm(self.log["step"], dim=1)
            grad_norm = torch.linalg.norm(self.log["grad"], dim=1)

            # Relative chi2 change
            rel_change = torch.abs(chi2_current - chi2_next) / (torch.abs(chi2_current) + 1e-10)
            converged_loss = rel_change < self.tol_loss
            # Absolute step size
            converged_step = step_norm < self.tol_step
            # Gradient norm relative to initial (scale-invariant)
            converged_grad = (grad_norm / (init_grad_norm + 1e-10)) < self.tol_grad

            converged = converged_loss | converged_step | converged_grad

            if torch.all(converged):
                waiting += 1
                if waiting >= self.patience:
                    if isPrint:
                        print(f"  >>> Converged after {iterations} iterations (patience={self.patience})", flush=True)
                    break
            else:
                waiting = 0

            # Global stall detection: if no batch accepted, apply perturbation after stall_thresh
            if accepted.any():
                global_stall = 0
            else:
                global_stall += 1
            stall_thresh = self.config.get("stall_perturb_thresh", 15)
            if global_stall >= stall_thresh:
                if isPrint:
                    print(f"  >>> Global stall detected at iter {iterations}, applying parameter perturbation", flush=True)
                # Apply adaptive perturbation scaled to per-parameter magnitude
                base_scale = self.config.get("perturb_scale", 1e-4)
                param_scale = x0.abs().clamp(min=1e-6).mean(dim=0, keepdim=True)
                # Blend: global scale for stability, per-param scale for relevance
                perturb = torch.randn_like(x0) * param_scale * base_scale
                # Do not perturb frozen parameters
                if getattr(self, "_frozen", False):
                    perturb = perturb * (~self._freeze_mask).float()
                x0 = x0 + perturb
                self._apply_param_bounds(x0)
                # Reset damping for all batches
                reset_val = torch.full((Nb,), self.init_damping * 10, device=self.device, dtype=self.dtype)
                damping = torch.clamp(reset_val, min=self.min_damping, max=self.max_damping)
                self.damping = damping
                self._stuck_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                self._recovery_count = torch.zeros(Nb, device=self.device, dtype=torch.int)
                global_stall = 0

            self.log["chi2_history"].append(chi2_next.mean().item())

            if isPrint:
                self._print_iter(iterations, chi2_current, time0,
                                 grad_norm, step_norm, damping, Nb, improved)

        # Return the best solution found (not just the last)
        # non-monotonic acceptance means last x0 may not be optimal
        # Final guard: compare best_x vs x0, keep whichever has lower chi2
        with torch.no_grad():
            chi2_best = self._Chi2(Y, self.forward(best_x), sig)
            chi2_last = self._Chi2(Y, self.forward(x0), sig)
            final_mask = (chi2_last < chi2_best)
            best_x = torch.where(final_mask[:, None], x0, best_x)
        self._apply_param_bounds(best_x)
        return best_x

    def _print_header(self, Nb, Ny, Nx, iterations, init_chi2, time0, damping):
        print(f"{'':=^100}", flush=True)
        print(f"{'Levenberg-Marquardt Optimization':^100}", flush=True)
        print(f"{'':=^100}", flush=True)
        print(f"  max_iters:{self.max_iters}  init_damping:{self.init_damping:.4e}", flush=True)
        print(f"  damping_inc:{self.damping_factor_increase}  damping_dec:{self.damping_factor_decrease}", flush=True)
        print(f"  damping_mode:{self.damping_mode}  damping_adapt:{self.damping_adapt}", flush=True)
        print(f"  trust_region:{self.trust_region}  grad_precond:{self.grad_precond}", flush=True)
        print(f"  solver:{self.step_solver}  batch:{Nb}  Ny:{Ny}  Nx:{Nx}  line_search:{self.max_line_search}", flush=True)
        print(f"{'':=^100}", flush=True)
        hdr = f" {'iter':<6s} | {'chi2':^16s} | {'time [min]':^16s} | {'grad':^16s} | {'step':^16s} | {'damping':^16s} | {'improved':^10s}"
        print(hdr, flush=True)
        msg = f" {iterations:<6d} | {init_chi2.mean().item():16.6e} | {(time.time()-time0)/60:^16.4f} | "
        msg += f"{'---':^16s} | {'---':^16s} | {damping.mean().item():16.6e} | {'---':^10s}"
        print(msg, flush=True)

    def _print_iter(self, iterations, chi2_current, time0, grad_norm, step_norm, damping, Nb, improved):
        n_imp = improved.sum().item()
        msg = f" {iterations:<6d} | {chi2_current.mean().item():16.6e} | {(time.time()-time0)/60:^16.4f} | "
        msg += f"{grad_norm.mean().item():16.6e} | {step_norm.mean().item():16.6e} | "
        msg += f"{damping.mean().item():16.6e} | {n_imp}/{Nb}"
        print(msg, flush=True)

    def lbfgs_refine(self, Y, x_start, sig=None, lr=1.0, max_iter=50,
                     history_size=20, tolerance_grad=1e-9,
                     tolerance_change=1e-13, line_search_fn="strong_wolfe",
                     isPrint=True):
        """Refine LM result using multi-stage L-BFGS with perturbation restart.

        L-BFGS is a quasi-Newton method that approximates the inverse Hessian
        from gradient history. Multi-stage approach with perturbation restart
        helps escape local minima that the initial LM solution may be trapped in.

        Stages:
          1. Coarse stage: moderate tolerance, larger history, warm-up
          2. Fine stage: tight tolerance, small step, precise convergence
          3. Perturbation restart: if fine stage did not improve enough,
             apply small perturbation and re-run fine stage

        Args:
            Y: Target data, shape (B, Ny)
            x_start: Starting params from LM, shape (B, Nx)
            sig: Noise covariance for chi2 weighting, shape (B, Ny)
            lr: Learning rate (default 1.0)
            max_iter: Max LBFGS iterations per stage (default 50)
            history_size: LBFGS memory (default 20)
            tolerance_grad: Gradient norm convergence tol
            tolerance_change: Param change convergence tol
            line_search_fn: "strong_wolfe" or None

        Returns:
            Refined parameters x, shape (B, Nx)
        """
        if sig is None:
            sig = torch.ones_like(Y)

        device = x_start.device
        dtype = x_start.dtype

        x = x_start.detach().clone()
        # Clamp microturbulence (index -2) and macroturbulence (index -1) to non-negative
        with torch.no_grad():
            self._apply_param_bounds(x)
        x.requires_grad_(True)

        # Build regularization terms from config (consistent with LM regularization)
        _reg_idx_lbfgs = self.config.get("reg_indices", [])
        _reg_tgt_lbfgs = self.config.get("reg_targets", [])
        _reg_str_lbfgs = self.config.get("reg_strengths", [])
        _has_reg_lbfgs = bool(_reg_idx_lbfgs) and len(_reg_idx_lbfgs)==len(_reg_tgt_lbfgs)==len(_reg_str_lbfgs)

        def batched_chi2_loss(x_param):
            y_pred = self.forward(x_param)
            loss = ((y_pred - Y).square() / sig).sum()
            if _has_reg_lbfgs:
                for k, idx in enumerate(_reg_idx_lbfgs):
                    s = float(_reg_str_lbfgs[k])
                    t = float(_reg_tgt_lbfgs[k])
                    loss = loss + s * (x_param[:, idx] - t).square().sum()
            # --- Add soft bounds barrier penalty so L-BFGS gradients respect bounds ---
            bounds_pen = self._compute_bounds_penalty(x_param)
            loss = loss + bounds_pen.sum()
            return loss

        def closure():
            if x.grad is not None:
                x.grad.zero_()
            # Clamp microturbulence (index -2) to non-negative
            with torch.no_grad():
                self._apply_param_bounds(x.data)
            loss = batched_chi2_loss(x)
            loss.backward()
            return loss

        with torch.no_grad():
            init_loss = batched_chi2_loss(x).item()

        if isPrint:
            print(f"  L-BFGS Multi-stage Refine:", flush=True)
            print(f"    init_loss={init_loss:.6e}", flush=True)

        t0 = time.time()

        # Stage 1: Coarse optimization with larger history
        coarse_iter = max(20, max_iter // 2)
        opt1 = torch.optim.LBFGS(
            [x], lr=lr, max_iter=coarse_iter,
            history_size=min(history_size * 2, 100),
            tolerance_grad=1e-6,
            tolerance_change=1e-10,
            line_search_fn=line_search_fn,
        )
        try:
            opt1.step(closure)
        except Exception as e:
            if isPrint:
                print(f"    Stage 1 early stop: {e}", flush=True)

        # Re-clamp in case L-BFGS line search moved parameters out of valid range
        with torch.no_grad():
            self._apply_param_bounds(x)
            loss1 = batched_chi2_loss(x).item()

        if isPrint:
            red1 = (init_loss - loss1) / init_loss * 100 if init_loss > 0 else 0
            print(f"    Stage 1 (coarse): loss={loss1:.6e}  reduction={red1:.2f}%", flush=True)

        # Stage 2: Fine optimization with tight tolerance
        opt2 = torch.optim.LBFGS(
            [x], lr=lr * 0.5, max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )
        try:
            opt2.step(closure)
        except Exception as e:
            if isPrint:
                print(f"    Stage 2 early stop: {e}", flush=True)

        with torch.no_grad():
            self._apply_param_bounds(x)
            loss2 = batched_chi2_loss(x).item()

        if isPrint:
            red2 = (loss1 - loss2) / loss1 * 100 if loss1 > 0 else 0
            print(f"    Stage 2 (fine):   loss={loss2:.6e}  reduction={red2:.2f}%", flush=True)

        # Stage 3: Perturbation restart if improvement is small
        perturb_thresh = self.config.get("lbfgs_perturb_thresh", 0.03)
        if loss1 > 0 and (loss1 - loss2) / loss1 < perturb_thresh:
            if isPrint:
                print(f"    Stage 3 (perturb): small improvement, applying perturbation restart", flush=True)

            x_best = x.detach().clone()
            loss_best = loss2

            # Per-parameter perturbation scaled to parameter magnitude
            with torch.no_grad():
                param_scale = x.abs().clamp(min=1e-6).mean(dim=0, keepdim=True)
                perturb_scale = self.config.get("lbfgs_perturb_scale", 0.01)
                perturb = torch.randn_like(x) * param_scale * perturb_scale
                x.add_(perturb)

            opt3 = torch.optim.LBFGS(
                [x], lr=lr * 0.3, max_iter=max_iter,
                history_size=history_size,
                tolerance_grad=tolerance_grad * 0.1,
                tolerance_change=tolerance_change * 0.1,
                line_search_fn=line_search_fn,
            )
            try:
                opt3.step(closure)
            except Exception as e:
                if isPrint:
                    print(f"    Stage 3 early stop: {e}", flush=True)

            with torch.no_grad():
                self._apply_param_bounds(x)
                loss3 = batched_chi2_loss(x).item()

            if loss3 > loss_best:
                if isPrint:
                    print(f"    Stage 3 (perturb): reverted to best (loss={loss_best:.6e})", flush=True)
                x = x_best
            else:
                if isPrint:
                    red3 = (loss2 - loss3) / loss2 * 100 if loss2 > 0 else 0
                    print(f"    Stage 3 (perturb): loss={loss3:.6e}  reduction={red3:.2f}%", flush=True)

        elapsed = time.time() - t0

        with torch.no_grad():
            final_loss = batched_chi2_loss(x).item()

        if isPrint:
            total_red = (init_loss - final_loss) / init_loss * 100 if init_loss > 0 else 0
            print(f"    Total: final_loss={final_loss:.6e}  reduction={total_red:.2f}%  time={elapsed:.1f}s", flush=True)

        return x.detach()


    def refine_groups(self, Y, x_start, active_groups, sig=None, max_iters=30,
                      isPrint=True, **kwargs):
        """Run additional LM iterations on specific parameter groups.

        Freezes all groups except active_groups and runs additional iterations
        to fine-tune selected groups. Useful for post-optimization refinement
        of parameters with weak signals (e.g., magnetic field B).
        """
        if sig is None:
            sig = torch.ones_like(Y[0])
        Nb, Ny = Y.shape
        Nx = x_start.size(1)
        device = x_start.device
        dtype = x_start.dtype
        dec = self.config.get("decomposition", [Nx])
        cum = [0] + list(np.cumsum(dec))
        ng = len(dec)
        fm = torch.ones(Nx, device=device, dtype=torch.bool)
        for g in active_groups:
            if g < ng:
                fm[cum[g]:cum[g+1]] = False
        na = int(fm.numel() - fm.sum().item())
        sf = getattr(self, "_frozen", False)
        sfm = getattr(self, "_freeze_mask", None)
        sl = getattr(self, "log", {})
        self._frozen = True
        self._freeze_mask = fm
        self.log = {}
        x0 = x_start.detach().clone().to(device).requires_grad_(False)
        self._apply_param_bounds(x0)
        Y = Y.to(device)
        st = sig.to(device) if torch.is_tensor(sig) else torch.tensor(sig, device=device, dtype=dtype)
        damp = torch.full((Nb,), self.init_damping, device=device, dtype=dtype)
        if self.adaptive_init_damping:
            _, _, _, Hi = self._LMcoef(Y, x0, sig=st, **kwargs)
            with torch.no_grad():
                adamp = self._compute_adaptive_init_damping(Hi)
                better = adamp > self.init_damping
                damp = torch.where(better, adamp, torch.full_like(damp, self.init_damping))
                x0 = x0.detach().clone().requires_grad_(False)
        self.damping = damp
        self._stuck_count = torch.zeros(Nb, device=device, dtype=torch.int)
        self._recovery_count = torch.zeros(Nb, device=device, dtype=torch.int)
        with torch.no_grad():
            i0 = self._check_losses(self._Chi2(Y, self.forward(x0), st), Nb)
        bx = x0.clone()
        bc = i0.clone()
        t0 = time.time()
        gn = ["T", "vLOS", "B", "gamma", "phi", "xi", "zeta"]
        an = [gn[g] for g in active_groups if g < ng]
        if isPrint:
            fs = ", ".join(an)
            fc = int(fm.sum().item())
            print("=" * 80, flush=True)
            print(f"  Parameter Group Refinement: [{fs}]  free={na}  frozen={fc}", flush=True)
            print(f"  max_iters={max_iters}  init_damping={self.init_damping:.4e}", flush=True)
            print("=" * 80, flush=True)
            h = f"  {'iter':>4s} | {'chi2':>16s} | {'time[s]':>8s} | {'damping':>12s} | {'ok':>6s}"
            print(h, flush=True)
            mi = float(i0.mean().item())
            md = float(damp.mean().item())
            print(f"  {0:>4d} | {mi:>16.6e} | {0:>8.4f} | {md:>12.6e} | {'---':>6s}", flush=True)
        iters = 0
        wait = 0
        pat = self.config.get("refine_patience", 5)
        for _ in range(max_iters):
            iters += 1
            x1 = self._Stepping(Y, x0, sig=st, **kwargs)
            cc = self.log["chi2_current"]
            cn = self._Chi2(Y, self.forward(x1), st).detach()
            impr = (cn < cc) & torch.isfinite(cn)
            acc = impr
            bx = torch.where((cn < bc)[:, None], x1.detach().clone(), bx)
            bc = torch.where(cn < bc, cn, bc)
            damp = self._adjust_damping(damp, impr, acc, cc, cn, torch.zeros_like(cc), iters, Y, x0, sig=st)
            self.damping = damp
            x1d = x1.detach().clone().requires_grad_(False)
            x0 = torch.where(acc[:, None], x1d, x0)
            self._apply_param_bounds(x0)
            sn = torch.linalg.norm(self.log["step"], dim=1)
            rc = torch.abs(cc - cn) / (torch.abs(cc) + 1e-10)
            cv = (rc < self.tol_loss) | (sn < self.tol_step)
            if bool(torch.all(cv)):
                wait += 1
                if wait >= pat:
                    if isPrint:
                        print(f"  >>> Refinement converged after {iters} iters", flush=True)
                    break
            else:
                wait = 0
            if isPrint:
                ni = int(impr.sum().item())
                mcc = float(cc.mean().item())
                md2 = float(damp.mean().item())
                et = time.time() - t0
                print(f"  {iters:>4d} | {mcc:>16.6e} | {et:>8.4f} | {md2:>12.6e} | {ni}/{Nb}", flush=True)
        self._frozen = sf
        if sfm is not None:
            self._freeze_mask = sfm
        self.log = sl
        el = time.time() - t0
        fc2 = self._check_losses(self._Chi2(Y, self.forward(bx), st), Nb)
        if isPrint:
            im = float(i0.mean().item())
            fm2 = float(fc2.mean().item())
            rd = (im - fm2) / im * 100 if im > 0 else 0
            print(f"  Refinement: {im:.6e} -> {fm2:.6e} ({rd:.2f}%)  [{el:.1f}s]", flush=True)
        return bx

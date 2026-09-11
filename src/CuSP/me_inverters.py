# -*- coding: utf-8 -*-
"""Batched LM / CMA-ES drivers for the ME inversion.

``CuSP.lm.BatchLM`` (Levenberg--Marquardt) and ``CuSP.cmaes.BatchCMAES``
(Covariance Matrix Adaptation Evolution Strategy) are general batched optimizers
built around a user-supplied ``forward(x) -> (M, Ny)`` callable and a per-element
noise variance ``sig`` for the chi2.  This module adapts them to the
Milne--Eddington problem:

* the parameter vector is the *normalised* 8-D vector that
  :class:`~CuSP.me_inversion.MEInversion` already uses (``[0, 1]^8``;
  ``denormalizing_parameter`` maps it to physics), so all three inversion
  methods (annealing, LM, CMA-ES) search exactly the same box and can share the
  same initial guess;
* the objective is the *same* weighted chi2 as
  ``MEInversion.merit_function`` (weights ``[1, 5, 5, 3.5]`` and sigmas
  ``[0.118, 0.204, 0.204, 0.204]`` for I, Q, U, V).  The optimizers are given the
  equivalent per-element variance through their ``sig`` argument, so their loss
  equals ``merit_function`` times the number of degrees of freedom
  ``4*N - 8``;
* the forward model is ``MEInversion.synthesize``, flattened to ``(M, 4*N)``
  with the channel order ``[I(N), Q(N), U(N), V(N)]``.

The parameters are clamped to a slightly wider box than ``[0, 1]`` before being
denormalised, which keeps the objective finite if an unconstrained step (LM) or
an overshooting sample (CMA-ES) leaves the physical box; because the clamp is
inside the forward, the chi2 is exactly flat outside the box.
"""

from __future__ import annotations

import numpy as np
import torch

from .cmaes import BatchCMAES
from .lm import BatchLM

__all__ = ["MEObjective", "run_lm", "run_cmaes"]


class MEObjective:
    """Flattened weighted-chi2 objective for one ``MEInversion`` + observation.

    Parameters
    ----------
    inversion : MEInversion
        Provides ``denormalizing_parameter`` and ``synthesize``.
    iquv_obs : Tensor (B, 4, N)
        Continuum-normalised observed Stokes vector.
    clamp : tuple(float, float) or None
        Box applied to the normalised parameters before denormalising
        (default ``(-1.0, 2.0)``; ``None`` disables it).

        The margin matters for LM: its unconstrained Newton steps need a finite
        gradient *just outside* the physical box to come back, so a strict
        ``(0, 1)`` clamp makes it stall (measured: from the PI2NN start LM
        reaches ``chi2/dof = 3.7e-13`` with ``(-1, 2)`` but only ``3.5e-5`` with
        ``(0, 1)``).  A wide margin, on the other hand, lets LM slide all the way
        to a mathematically exact but *unphysical* solution; the caller
        (``MEInversion.__call__``) therefore projects the result back into
        ``[0, 1]`` and keeps the better of the projection and the initial guess.
    """

    def __init__(self, inversion, iquv_obs: torch.Tensor, clamp=(-1.0, 2.0)):
        obs = iquv_obs.detach()
        self.inversion = inversion
        self.device = obs.device
        self.dtype = obs.dtype
        self.n_lambda = int(obs.shape[2])
        self.n_stokes = int(obs.shape[1])
        self.Y = obs.reshape(obs.shape[0], -1)                    # (B, 4N)
        # per-element variance, expanded to (B, 4N): BatchCMAES expands the
        # batch axis itself (sig.unsqueeze(1).expand(-1, pop, -1)), so a
        # broadcastable (1, 4N) row only works for B == 1
        self.sig = inversion.measurement_variance(self.n_lambda).to(
            device=self.device, dtype=self.dtype).expand(self.Y.shape[0], -1)
        self.dof = int(self.Y.shape[1]) - 8
        if self.dof <= 0:
            raise ValueError(
                f"inversion DOF <= 0: 4*N - 8 = {self.dof} for N = {self.n_lambda}"
            )
        self.clamp = clamp

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised parameters ``(M, 8)`` -> flattened Stokes ``(M, 4N)``."""
        if self.clamp is not None:
            x = x.clamp(min=self.clamp[0], max=self.clamp[1])
        phys = self.inversion.denormalizing_parameter(x)
        return self.inversion.synthesize(phys).reshape(x.shape[0], -1)

    def chi2(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted chi2 ``(M,)`` of the normalised parameters ``x``."""
        with torch.no_grad():
            resid = self.Y[: x.shape[0]] - self.forward(x)
            return (resid.square() / self.sig).sum(dim=1)

    def merit(self, x: torch.Tensor) -> torch.Tensor:
        """``chi2 / dof``, i.e. the quantity ``MEInversion.merit_function`` returns."""
        return self.chi2(x) / self.dof

    def describe(self) -> str:
        return (f"MEObjective: {self.Y.shape[0]} pixel(s) x {self.Y.shape[1]} observables, "
                f"dof={self.dof}, lambda samples={self.n_lambda}")


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def run_lm(objective: MEObjective, x_guess: torch.Tensor, max_iters: int = 60,
           isPrint: bool = False, config: dict | None = None, **options):
    """Run :class:`CuSP.lm.BatchLM` on the ME objective.

    Returns ``(x_best, optimizer)``; ``x_best`` is in normalised parameter space.
    """
    n_param = int(x_guess.size(1))
    # decomposition=[n_param] keeps BatchLM's pyPRT-specific per-group bound logic
    # (vLos in km/s, gamma/phi in [0, pi], ...) inert; the ME box is handled by
    # MEObjective.clamp instead.
    cfg = dict(decomposition=[n_param])
    if config:
        cfg.update(config)
    optimizer = BatchLM(max_iters=max_iters, forward=objective.forward, **options)
    x_best = optimizer(objective.Y, x_guess, sig=objective.sig, config=cfg,
                       isPrint=isPrint, device=objective.device, dtype=objective.dtype)
    # BatchLM's own "best" bookkeeping is reset whenever its stall-recovery
    # perturbes the current point (lm.py resets best_chi2 to the perturbed x0
    # without resetting best_x), so the returned point can be worse than the
    # starting guess.  Keep the better of the two, which is what the caller
    # expects from an optimizer.
    with torch.no_grad():
        chi2_new = objective.chi2(x_best.to(objective.device, objective.dtype))
        chi2_start = objective.chi2(x_guess.to(objective.device, objective.dtype))
        x_best = torch.where((chi2_new < chi2_start)[:, None], x_best, x_guess)
    return x_best, optimizer


def run_cmaes(objective: MEObjective, x_guess: torch.Tensor, max_iters: int = 200,
              pop_size: int | None = None, init_sigma: float = 0.25,
              patience: int = 50, bounded: bool = True, isPrint: bool = False,
              config: dict | None = None, **options):
    """Run :class:`CuSP.cmaes.BatchCMAES` on the ME objective.

    ``bounded=True`` (default) keeps the search inside the ``[0, 1]`` box through
    the class's sigmoid/logit bounds transform.  Returns ``(x_best, optimizer)``
    with ``x_best`` in normalised parameter space.
    """
    n_param = int(x_guess.size(1))
    cfg = dict(decomposition=[n_param], scalings=["lin"])
    if bounded:
        cfg["bounds"] = {0: [0.0, 1.0]}
    if config:
        cfg.update(config)
    optimizer = BatchCMAES(max_iters=max_iters, pop_size=pop_size,
                           init_sigma=init_sigma, patience=patience,
                           forward=objective.forward, **options)
    x_best = optimizer(objective.Y, x_guess, sig=objective.sig, config=cfg,
                       isPrint=isPrint, device=objective.device, dtype=objective.dtype)
    return x_best, optimizer

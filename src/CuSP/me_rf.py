# -*- coding: utf-8 -*-
"""
Vectorised response functions (Jacobians) of the ME forward model.

The ME forward model is *element-wise* in every (batch, wavelength) pair: the
Stokes vector at one wavelength depends only on the eight ME parameters of that
pixel, never on the neighbouring wavelengths.  Differentiating it the classical
way -- one `torch.autograd.grad` per observable, i.e. ``4 * Nw`` backward passes
plus one for the loss gradient -- therefore scales linearly with the number of
wavelength samples, which is pure waste when the observed spectrum is sampled
densely (a 100-point spectrum needs 400 backward passes).

This module exploits the element-wise structure instead:

1. the parameters ``x`` of shape ``(Nb, 8)`` are expanded to
   ``x_exp = x[:, None, :].expand(Nb, Nw, 8)`` and turned into an autograd leaf,
   so every ``(batch, wavelength, parameter)`` entry is an independent variable;
2. the forward model is evaluated for all ``Nb * Nw`` elements at once, giving
   Stokes ``(Nb, Nw, 4)``;
3. four backward passes -- one per Stokes component -- give the response
   functions ``RF_{I,Q,U,V}`` of shape ``(Nb, Nw, 8)``, which are stacked into
   ``(Nb, Nw, 4, 8)`` and, for LM, permuted into the ``(Nb, 4*Nw, 8)`` Jacobian.

The cost is thus ``O(1)`` backward passes *plus* one extra pass for the
continuum normalisation, independent of ``Nw``.  The normalisation must be
handled explicitly: the forward model divides the line Stokes vector by its own
Stokes-I value at the continuum wing, and that wing value depends on the same
eight parameters, so its derivative is folded in with the quotient rule (see
``__call__``).  To keep the graph small the wing is carried as element 0 of the
expanded wavelength axis rather than being recomputed per element.

Both the element-wise forward (``profile_from_u`` / ``stokes_from_eta_rho``,
reached through the public helpers of :class:`CuSP.me_forward.MEForward`) and
the line-profile convention are shared with the reference forward model, so the
response functions here are the exact derivatives of ``MEForward.return_IQUV``.
"""

import torch

from .voigt import VoigtProfile, VoigtFaradayProfile  # noqa: F401  (same convention)

__all__ = [
    'VectorizedResponseFunction',
    'vectorized_response_function',
    'make_vectorized_lmcoef',
]

LAYOUTS = ('nw48', 'lm')
N_STOKES = 4


class VectorizedResponseFunction:
    """Response function ``d(Stokes)/d(parameters)`` of one ME forward model.

    Parameters
    ----------
    forward : MEForward
        Provides ``wavebands`` (continuum wing in slot 0, then the sample
        points), ``lambda0``, ``G``, ``denormalizing_parameter`` and the physics
        helpers.  An :class:`CuSP.me_inversion.MEInversion` works as well.
    wavelengths : Tensor, optional
        Sample wavelengths in nm; defaults to ``forward.wavebands[1:]``.
    wing : Tensor or float, optional
        Continuum wavelength used for the normalisation; defaults to
        ``forward.wavebands[0]``.
    zoom_factor : float
        Multiplies the differentiated output and divides it back out, i.e. a
        mathematically exact rescaling kept for interface parity with
        ``CuSP.lm.BatchLM`` (which uses 1e-15).  The element-wise graph needs no
        such trick, so the default here is 1.0.

    Notes
    -----
    ``__call__`` returns the normalised-Stokes response function, i.e. the
    derivative of exactly what ``MEForward.return_IQUV`` (and therefore
    ``MEInversion.synthesize``) produces, including the derivative of the
    continuum normalisation.
    """

    def __init__(self, forward, wavelengths=None, wing=None, zoom_factor=1.0):
        self.forward = forward
        ref = forward.wavebands
        if wavelengths is None:
            wavelengths = ref[1:]
        else:
            wavelengths = torch.as_tensor(wavelengths).to(ref)
        if wing is None:
            wing = ref[:1]
        else:
            wing = torch.as_tensor(wing).reshape(1).to(ref)
        self.wavelengths = wavelengths.reshape(-1)
        self.wing = wing.reshape(1)
        self.zoom_factor = zoom_factor

    # ------------------------------------------------------------------ #
    @property
    def n_wavelengths(self) -> int:
        return int(self.wavelengths.numel())

    @property
    def n_params(self) -> int:
        return 8

    def grid(self) -> torch.Tensor:
        """Wing first, then the sample wavelengths -> ``(Nw+1,)``."""
        return torch.cat([self.wing, self.wavelengths])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(Nw={self.n_wavelengths}, "
                f"lambda0={getattr(self.forward, 'lambda0', None)}, "
                f"G={getattr(self.forward, 'G', None)})")

    # ------------------------------------------------------------------ #
    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """``(Nb, 8)`` -> autograd leaf ``(Nb, Nw+1, 8)``.

        The wing occupies element 0 and the samples elements ``1..Nw``; every
        element is an independent variable that still holds the same parameter
        values, which is what makes the per-wavelength derivatives separable
        with a single backward pass per Stokes component.
        """
        Nb, Nx = x.shape
        x_exp = x.detach().reshape(Nb, 1, Nx).expand(Nb, self.n_wavelengths + 1, Nx)
        return x_exp.clone().requires_grad_(True)

    def _elementwise_stokes(self, x_exp: torch.Tensor, normalized: bool = True):
        """Forward model on the expanded grid.

        Returns ``(stokes, wing_I)`` with ``stokes`` the normalised Stokes vector
        ``(Nb, Nw, 4)`` at the sample wavelengths and ``wing_I`` its Stokes-I
        normalisation ``(Nb, 1, 1)`` at the wing.
        """
        Nb, Nall, Nx = x_exp.shape
        if Nx != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {Nx}")
        flat = x_exp.reshape(Nb * Nall, Nx)
        phys = self.forward.denormalizing_parameter(flat) if normalized else flat
        phys = phys.reshape(Nb, Nall, self.n_params)
        Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi = (
            phys[..., i:i + 1] for i in range(self.n_params))

        u_los = (self.forward.lambda0 * v_los / self.forward.c) / Dlambda_D
        u_B = (4.67e-12 * self.forward.lambda0 ** 2 * Bmag) / Dlambda_D
        lam = self.grid().to(x_exp).reshape(1, Nall, 1)
        u0 = (lam - self.forward.lambda0) / Dlambda_D

        profiles = self.forward.profile_from_u(u0, u_los, u_B, a_damp)
        eta_rho = self.forward.return_eta_rho(eta_0, theta, phi, *profiles)
        tau0 = torch.cat(self.forward.stokes_from_eta_rho(*eta_rho, S10), dim=-1)
        wing_I = tau0[:, 0, 0].reshape(Nb, 1, 1)
        stokes = tau0[:, 1:, :] / wing_I
        return stokes, wing_I

    # ------------------------------------------------------------------ #
    def __call__(self, x, normalized: bool = True, layout: str = 'nw48',
                 return_stokes: bool = False):
        """Response function of ``x``.

        Parameters
        ----------
        x : Tensor ``(Nb, 8)`` or ``(8,)``
            Parameters, normalised to ``[0, 1]`` by default (the optimizer
            convention); pass ``normalized=False`` for physical parameters.
        normalized : bool
            Whether ``x`` is the normalised parameter vector.
        layout : {'nw48', 'lm'}
            ``'nw48'`` -> ``(Nb, Nw, 4, 8)`` (Stokes component before the
            parameter axis); ``'lm'`` -> ``(Nb, 4*Nw, 8)`` with the channels
            flattened in the same I,Q,U,V-major order as
            ``MEInversion.synthesize(...).reshape(Nb, -1)``, i.e. exactly the
            Jacobian ``CuSP.lm.BatchLM`` expects.
        return_stokes : bool
            Also return the normalised Stokes spectra ``(Nb, Nw, 4)`` computed on
            the same graph (so no second forward pass is needed).

        Returns
        -------
        rf, or ``(stokes, rf)`` when ``return_stokes`` is True.
        """
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        x = torch.as_tensor(x)
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.size(1) != self.n_params:
            raise ValueError(f"x must be (Nb, {self.n_params}), got {tuple(x.shape)}")
        Nb = x.size(0)

        x_exp = self._expand(x)
        stokes, wing_I = self._elementwise_stokes(x_exp, normalized=normalized)

        zoom = self.zoom_factor
        # One backward pass per Stokes component.  `grad[:, 1:, :]` is the
        # derivative of the line profile w.r.t. its *own* element's parameters;
        # the wing (element 0) enters only through the normalisation, which the
        # quotient rule below removes explicitly.
        rf = []
        for k in range(N_STOKES):
            out = stokes[..., k] * zoom
            grad = torch.autograd.grad(
                outputs=out,
                inputs=x_exp,
                grad_outputs=torch.ones_like(out),
                create_graph=False,
                retain_graph=True,
            )[0]
            rf.append(grad[:, 1:, :].detach() / zoom)
        wing_out = wing_I * zoom
        wing_grad = torch.autograd.grad(
            outputs=wing_out,
            inputs=x_exp,
            grad_outputs=torch.ones_like(wing_out),
            create_graph=False,
            retain_graph=False,
        )[0]
        # f = N / D  ->  df/dp = (dN/dp - f * dD/dp) / D
        rf = torch.stack(rf, dim=2)                                  # (Nb, Nw, 4, 8)
        gD = wing_grad[:, 0:1, :].detach() / zoom / wing_I.detach()  # (Nb, 1, 8)
        rf = rf - stokes.detach().unsqueeze(-1) * gD.unsqueeze(2)    # (Nb, Nw, 4, 8)
        rf = rf.detach()

        if layout == 'lm':
            rf = rf.permute(0, 2, 1, 3).reshape(Nb, 4 * self.n_wavelengths, self.n_params)
        if squeeze:
            rf = rf[0]
            stokes = stokes[0]
        if return_stokes:
            return stokes.detach(), rf
        return rf


def vectorized_response_function(forward, x, normalized: bool = True,
                                 layout: str = 'nw48', **kwargs):
    """Convenience wrapper: response function of ``x`` for ``forward``.

    ``kwargs`` (``wavelengths``, ``wing``, ``zoom_factor``) are forwarded to
    :class:`VectorizedResponseFunction`.
    """
    return VectorizedResponseFunction(forward, **kwargs)(x, normalized=normalized,
                                                         layout=layout)


def make_vectorized_lmcoef(objective, response=None, zoom_factor: float = 1.0,
                           penalty=None):
    """Build a drop-in replacement for ``CuSP.lm.BatchLM._LMcoef``.

    The returned callable keeps BatchLM's contract
    ``(Y, x, **kwargs) -> (chi2, grad, Jacobian, Hessian)`` where ``Hessian`` is
    the Gauss-Newton matrix ``J^T W J`` (``W = 1/sig``), but obtains ``J`` from
    :class:`VectorizedResponseFunction` instead of one backward pass per
    observable.  ``BatchLM`` accepts it through its ``usrLMcoef`` hook, which is
    how ``CuSP.me_inverters.run_lm`` installs it as the default.

    ``Y`` is the flattened observation ``(Nb, 4*Nw)`` in I,Q,U,V-major order and
    ``sig`` (keyword) its per-element variance; both default to the objective's.
    The gradient of the weighted chi2 is taken from the response function by the
    chain rule (``dchi2/dp = -2 sum_i r_i/sig_i * J_ip``), which is exact and
    costs no extra backward pass.

    ``objective.clamp`` (if not ``None``) is applied exactly like
    ``MEObjective.forward`` does, and the Jacobian columns of parameters that sit
    outside the box are zeroed, reproducing the derivative of the clamped
    objective (a parameter pinned at the margin has no gradient, as in the
    loop-over-observables implementation).

    Parameters
    ----------
    penalty : callable, optional
        Extra differentiable term added to chi2 (mirroring BatchLM's soft bounds
        penalty).  ``run_lm`` configures ``decomposition=[8]``, for which that
        term is identically zero, so it is not used by default.
    """
    inversion = objective.inversion
    if response is None:
        response = VectorizedResponseFunction(inversion, zoom_factor=zoom_factor)
    clamp = getattr(objective, 'clamp', None)
    n_lambda = int(objective.n_lambda)
    n_obs = 4 * n_lambda

    def lmcoef(Y, x, **kwargs):
        sig = kwargs.get('sig', None)
        if sig is None:
            sig = objective.sig
        x = x.detach().to(device=Y.device, dtype=Y.dtype)
        sig = sig.detach().to(device=Y.device, dtype=Y.dtype)
        Nb, Ny = Y.shape
        if Ny != n_obs:
            raise ValueError(
                f"objective expects {n_obs} observables (4 x {n_lambda} wavelengths), "
                f"got {Ny}")
        if x.size(0) != Nb:
            raise ValueError(f"batch mismatch: Y has {Nb} rows, x has {x.size(0)}")

        inside = None
        if clamp is not None:
            lo, hi = float(min(clamp)), float(max(clamp))
            inside = (x >= lo) & (x <= hi)
            x = x.clamp(min=lo, max=hi)
        if sig.size(0) != Nb:
            sig = sig.expand(Nb, -1)

        stokes, J = response(x, normalized=True, layout='lm', return_stokes=True)
        if inside is not None:
            J = J * inside.unsqueeze(1)
        spectra = stokes.permute(0, 2, 1).reshape(Nb, n_obs)

        resid = Y - spectra
        chi2 = (resid.square() / sig).sum(dim=1)
        if penalty is not None:
            chi2 = chi2 + penalty(x)
        grad = -(2.0 * resid / sig).unsqueeze(1).matmul(J).squeeze(1)
        Hessian = torch.einsum("bni,bnj->bij", J / sig.unsqueeze(-1), J)
        return (chi2.detach(), grad.detach(), J.detach(), Hessian.detach())

    return lmcoef

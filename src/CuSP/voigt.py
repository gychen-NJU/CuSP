# -*- coding: utf-8 -*-
"""Fast Voigt / Faraday-Voigt line profiles for CuSP.

This module replaces CuSP's original fixed-grid trapezoidal quadrature with the
complex rational approximation of the Faddeeva function used by
``codes/voigt.py`` (``voigt_profiles``), which needs no numerical integration.

Conventions are **identical to the original CuSP routines** (so nothing
downstream changes):

    phi(u, a) = Re w(u + i a) / sqrt(pi)      with  int phi du = 1
    psi(u, a) = Im w(u + i a) / sqrt(pi)

and both keep the original call signature ``f(u, a, ynodes=..., lim=...)`` and
the original output layout: the grid axis of the old quadrature is *reduced* by
``trapz``, so for ``u`` of shape ``(N, Nw)`` and ``a`` of shape ``(N, 1)`` the
result has shape ``(N, Nw)`` -- i.e. exactly the broadcast shape of ``(u, a)``.

Implementation notes
--------------------
``codes/voigt.py`` evaluates the 7/7 rational function ``P/Q`` at ``z = a - i u``.
It was verified numerically (see ``analysis/probe_rational_domain.py``,
``analysis/compare_voigt.py``) that this equals the Faddeeva function at the
"physical" argument:

    P(a - i u) / Q(a - i u)  ==  w(u + i a)     (|W - w| <= 1e-7 for |u| <= 600)

so the profiles follow directly as ``Re W / sqrt(pi)`` and ``Im W / sqrt(pi)``.
Because ``Re W`` is even and ``Im W`` is odd in ``u``, the sign-folding used by
``codes/voigt.py`` is algebraically redundant and is not needed here.

Accuracy against ``scipy.special.wofz`` (measured, absolute error of ``phi`` /
``psi``, ``|u| <= 200``)::

    a = 0      2.5e-6 / 2.5e-6      (the old quadrature returned exactly 0 !)
    a = 1e-4   2.5e-6 / 2.5e-6
    a = 0.01   2.4e-6 / 2.4e-6
    a = 0.1    1.4e-6 / 1.5e-6
    a = 0.5    2.2e-7 / 2.1e-7
    a = 1.0    3.6e-8 / 3.7e-8

The same accuracy holds in float32, so no dtype promotion is required.  The
``a -> 0`` Dawson/interpolation branch of ``codes/voigt.py`` is therefore not
needed (and would have required the missing ``codes/interpolation.py``).

Validity range: the polynomial is evaluated in Horner form on ``z``, so ``z**7``
overflows for ``|z| >~ 1.4e5`` in float32 (``>~ 1e43`` in float64) and ``inf/inf``
would give NaN.  CuSP's operating range is ``|u| = |lambda - lambda0| / Dlambda_D
<~ 600`` (``Dlambda_D >= 1e-4`` nm, ``|lambda - lambda0| <= 0.02`` nm, plus the
Zeeman/Doppler shifts), i.e. a factor ~200 inside the float32 limit.

The previous quadrature implementations are kept unchanged for reference and
cross-checks as :func:`VoigtProfileQuadrature` /
:func:`VoigtFaradayProfileQuadrature`.
"""

import math

import torch

__all__ = [
    "VoigtProfile",
    "VoigtFaradayProfile",
    "voigt_profiles",
    "faddeeva_rational",
    "VoigtProfileQuadrature",
    "VoigtFaradayProfileQuadrature",
    "VOIGT_RATIONAL_A",
    "VOIGT_RATIONAL_B",
]

# 7/7 rational approximation coefficients, as used by codes/voigt.py
VOIGT_RATIONAL_A = (
    122.607931777104326, 214.382388694706425, 181.928533092181549,
    93.155580458138441, 30.180142196210589, 5.912626209773153,
    0.564189583562615,
)
VOIGT_RATIONAL_B = (
    122.60793177387535, 352.730625110963558, 457.334478783897737,
    348.703917719495792, 170.354001821091472, 53.992906912940207,
    10.479857114260399,
)

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_COEFF_CACHE = {}


def _rational_coeffs(device, dtype):
    """(A6..A0, B6..B0) tensors on ``device``/``dtype``, cached."""
    key = (str(device), str(dtype))
    coeffs = _COEFF_CACHE.get(key)
    if coeffs is None:
        a = torch.tensor(VOIGT_RATIONAL_A, device=device, dtype=dtype)
        b = torch.tensor(VOIGT_RATIONAL_B, device=device, dtype=dtype)
        coeffs = (a[6], a[5], a[4], a[3], a[2], a[1], a[0],
                  b[6], b[5], b[4], b[3], b[2], b[1], b[0])
        _COEFF_CACHE[key] = coeffs
    return coeffs


def faddeeva_rational(z):
    """Faddeeva function ``w(z)`` by the 7/7 rational approximation.

    Parameters
    ----------
    z : torch.Tensor (complex)
        Argument.  For the line profiles below this is ``a - 1j * u``.

    Returns
    -------
    torch.Tensor (complex)
        Approximation of ``w(z)``, same shape/device as ``z``.
    """
    A6, A5, A4, A3, A2, A1, A0, B6, B5, B4, B3, B2, B1, B0 = _rational_coeffs(
        z.device, z.real.dtype
    )
    numerator = ((((((A6 * z + A5) * z + A4) * z + A3) * z + A2) * z + A1) * z + A0)
    denominator = ((((((z + B6) * z + B5) * z + B4) * z + B3) * z + B2) * z + B1) * z + B0
    return numerator / denominator


def _as_float_pair(u, a):
    """Bring (u, a) to a common floating dtype (complex ops need matching dtypes)."""
    dtype = torch.promote_types(u.dtype, a.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    if u.dtype != dtype:
        u = u.to(dtype)
    if a.dtype != dtype:
        a = a.to(dtype)
    return u, a


def voigt_profiles(u, a):
    """Normalised Voigt (``phi``) and Faraday-Voigt (``psi``) profiles.

    No numerical integration: the complex rational approximation of ``w(z)``
    is evaluated at ``z = a - 1j * u``.

    Parameters
    ----------
    u : torch.Tensor
        Normalised frequency/wavelength detuning; any shape that broadcasts
        with ``a`` (typically ``(N, Nw)``).
    a : torch.Tensor
        Damping constant; typically ``(N, 1)``.

    Returns
    -------
    phi, psi : torch.Tensor
        Both broadcast to the common shape of ``u`` and ``a``, same dtype as
        the (promoted) inputs, ``int phi du = 1``.
    """
    u, a = _as_float_pair(u, a)
    w = faddeeva_rational(torch.complex(a.to(u.device), -u))
    return w.real * _INV_SQRT_PI, w.imag * _INV_SQRT_PI


def VoigtProfile(u: torch.Tensor, a: torch.Tensor, ynodes=None, lim=None) -> torch.Tensor:
    """Voigt absorption profile ``phi(u, a)``.

    Drop-in replacement for the original CuSP function: same signature (the
    ``ynodes`` / ``lim`` grid parameters are accepted for backward
    compatibility and ignored) and the same result shape/device.

    ``u`` of shape ``(N, Nw)`` and ``a`` of shape ``(N, 1)`` give ``(N, Nw)``,
    exactly as the old quadrature did (``trapz`` reduced the y-grid axis).
    """
    phi, _ = voigt_profiles(u, a)
    return phi


def VoigtFaradayProfile(u: torch.Tensor, a: torch.Tensor, ynodes=None, lim=None) -> torch.Tensor:
    """Faraday-Voigt (anomalous dispersion) profile ``psi(u, a)``.

    Drop-in replacement for the original CuSP function; see
    :func:`VoigtProfile` for the shape/signature contract.
    """
    _, psi = voigt_profiles(u, a)
    return psi


# --------------------------------------------------------------------------- #
# Original quadrature implementations, kept for reference / cross-checks.
# --------------------------------------------------------------------------- #
_trapz = getattr(torch, "trapezoid", None) or torch.trapz


def VoigtFaradayProfileQuadrature(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=100,
    lim=5.0,
) -> torch.Tensor:
    """Original CuSP implementation: trapezoidal quadrature over a fixed y-grid."""
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim, lim, ynodes, device=device, dtype=dtype)
    dy = y[1] - y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None, None, :]
    numerator = torch.exp(-y ** 2) * (u - y)
    denominator = (u - y) ** 2 + a ** 2
    integrand = numerator / denominator
    profile = (1 / torch.pi ** 1.5) * _trapz(integrand, dx=dy, dim=-1)
    return profile


def VoigtProfileQuadrature(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=1000,
    lim=10.0,
) -> torch.Tensor:
    """Original CuSP implementation: trapezoidal quadrature over a fixed y-grid."""
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim, lim, ynodes, device=device, dtype=dtype)
    dy = y[1] - y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None, None, :]
    try:
        numerator = torch.exp(-y ** 2)
        denominator = (u - y) ** 2 + a ** 2
        integrand = numerator / denominator
        profile = (a[:, :, 0] / torch.pi ** 1.5) * _trapz(integrand, dx=dy, dim=-1)
    except Exception:
        print(u.shape, a.shape, y.shape)
        raise
    return profile

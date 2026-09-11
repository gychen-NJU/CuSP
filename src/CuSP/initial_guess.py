# -*- coding: utf-8 -*-
"""Initial guesses for the CuSP ME inversion.

CuSP's :class:`~CuSP.me_inversion.MEInversion` anneals in a normalised 8-D
parameter space and normally starts from a uniform random guess.  This module
adds *learned* initial guesses: a trained PI2NN network is evaluated on the
target Stokes spectrum and its output is used as the starting point, which is
what makes ME inversions of large HMI-like data sets tractable (the annealer
starts already close to the solution instead of exploring the full box).

Usage (see :class:`~CuSP.me_inversion.MEInversion`)::

    from CuSP import MEInversion

    inv = MEInversion(wavebands, landeG=2.5, lambda0=617.33352, wing=617.31714)
    params = inv(iquv_obs, initial_guess='sdo_hmi')

``initial_guess`` accepts

* ``None`` (default) -- uniform random guess, the original behaviour,
* ``'sdo_hmi'``      -- the shipped PI2NN network for SDO/HMI Fe I 6173 A,
* a registered name (see :func:`register_initial_guess`),
* a path to a ``.pkl`` produced by ``torch.save(PI2NN_instance, ...)``,
* a ``torch.Tensor`` of normalised parameters with shape ``(B, 8)``,
* a callable ``f(iquv_obs, inversion=...)`` returning physical or normalised
  parameters.

Models are loaded through :func:`load_pi2nn_model`, which resolves the pickle's
``PI2NN.*`` class references to :mod:`CuSP.PI2NN` (the training scripts were run
with ``PI2NN`` importable as a top level module, so a plain ``torch.load``
cannot unpickle them).
"""

from __future__ import annotations

import inspect
import io
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "load_pi2nn_model",
    "PI2NNInitialGuess",
    "get_initial_guess_model",
    "list_initial_guess_models",
    "register_initial_guess",
    "initial_guess_from_spectrum",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"

#: Pickle module names that must be remapped to :mod:`CuSP.PI2NN` (the training
#: scripts imported ``PI2NN`` as a top level module, and older ones ran as
#: ``__main__``).
_PI2NN_PICKLE_MODULES = ("PI2NN", "__main__")

_TORCH_LOAD_HAS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


# --------------------------------------------------------------------------- #
# shipped models
# --------------------------------------------------------------------------- #
@dataclass
class _ModelSpec:
    filename: str
    description: str


_REGISTRY: dict[str, _ModelSpec] = {
    "sdo_hmi": _ModelSpec(
        filename="pi2nn_sdo_hmi.pkl",
        description=(
            "PI2NN trained on synthetic SDO/HMI-like Fe I 6173 A spectra "
            "(6 wavelength samples, landeG=2.5, lambda0=617.33352 nm)"
        ),
    ),
}

_MODEL_CACHE: dict[str, "PI2NNInitialGuess"] = {}


def list_initial_guess_models() -> dict[str, str]:
    """Return ``{name: description}`` of the registered initial-guess models."""
    return {name: spec.description for name, spec in _REGISTRY.items()}


def register_initial_guess(name: str, path, description: str = "") -> None:
    """Register a PI2NN checkpoint (``.pkl``) under ``name``."""
    _REGISTRY[name] = _ModelSpec(filename=str(path), description=description or str(path))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _pi2nn_class_map() -> dict:
    """``{class name in pickle: real class}`` for the PI2NN module."""
    from . import PI2NN as _mod

    return {
        "PI2NN": _mod.PI2NN,
        "InversionNet": _mod.InversionNet,
        "FCN": _mod.FCN,
        "ResidualBlock": _mod.ResidualBlock,
        "ConvResidualBlock": _mod.ConvResidualBlock,
        "ChannelAttention": _mod.ChannelAttention,
        "SelfAttention": _mod.SelfAttention,
    }


def _make_unpickler(class_map: dict):
    """A :class:`pickle.Unpickler` that remaps the PI2NN module references."""

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in _PI2NN_PICKLE_MODULES:
                cls = class_map.get(name)
                if cls is not None:
                    return cls
            return super().find_class(module, name)

    return _Unpickler


class _PickleModuleShim:
    """Minimal ``pickle_module`` accepted by ``torch.load`` (see pyprt.utils.loaders).

    ``torch.load`` inspects ``pickle_module.__name__`` (dill check) and calls
    ``Unpickler`` / ``load`` / ``loads`` on it.
    """

    __name__ = "pickle"

    def __init__(self, unpickler):
        self.Unpickler = unpickler

    def load(self, file, **kwargs):
        return self.Unpickler(file, **kwargs).load()

    def loads(self, data, **kwargs):
        return self.Unpickler(io.BytesIO(data), **kwargs).load()


def load_pi2nn_model(path, map_location="cpu", device=None):
    """Load a PI2NN checkpoint saved by ``torch.save(PI2NN_instance, ...)``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the ``.pkl`` (or a registered model name, e.g. ``'sdo_hmi'``).
    map_location : str or torch.device
        Passed to ``torch.load``; where the stored tensors are restored.
    device : str or torch.device, optional
        Device to move the network to.  Defaults to ``map_location``.

    Returns
    -------
    CuSP.PI2NN.PI2NN
        The restored model.  ``model.device`` and the network parameters are put
        on ``device`` (the shipped checkpoint was stored from a CUDA run, so this
        also clears a stale ``cuda:1`` attribute) and the network is put in
        evaluation mode.
    """
    spec = _REGISTRY.get(str(path))
    file = Path(spec.filename) if spec is not None else Path(path)
    if spec is not None and not file.is_absolute() and not file.exists():
        file = _DATA_DIR / file.name
    if not file.exists():
        raise FileNotFoundError(
            f"PI2NN checkpoint not found: {file}"
            + (f" (registered name '{path}')" if spec is not None else "")
        )

    class_map = _pi2nn_class_map()
    pickle_module = _PickleModuleShim(_make_unpickler(class_map))
    kwargs = dict(map_location=map_location, pickle_module=pickle_module)
    if _TORCH_LOAD_HAS_WEIGHTS_ONLY:
        kwargs["weights_only"] = False
    model = torch.load(file, **kwargs)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"{file} does not contain a torch.nn.Module (got {type(model).__name__}); "
            "expected a full PI2NN instance saved with torch.save(model, path)"
        )

    target = torch.device(device if device is not None else map_location)
    model.net.to(target)
    model.device = target
    # NOTE: PI2NN overrides ``nn.Module.train`` with its training loop, and
    # ``nn.Module.eval()`` is implemented as ``self.train(False)`` -- calling
    # ``model.eval()`` here would *run the training loop* (generating a million
    # random spectra and writing PI2NN_models/*.npz).  Flip the flag directly.
    model.training = False
    model.net.eval()
    return model


# --------------------------------------------------------------------------- #
# using a model as an initial guess
# --------------------------------------------------------------------------- #
class PI2NNInitialGuess:
    """Wrap a trained PI2NN as an initial-guess generator for ME inversions.

    The network maps a continuum-normalised Stokes vector
    ``iquv_obs[B, 4, N]`` (channels ordered I, Q, U, V, as produced by
    :meth:`CuSP.me_forward.MEForward.return_IQUV`) to the 8 physical ME
    parameters ``(Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi)``.

    Attributes
    ----------
    lambda0, landeG, wing : float
        Configuration the network was trained with.  A spectrum computed with a
        different ``lambda0``/``landeG``/wavelength sampling must not be fed to
        the network (see :meth:`check_inversion`).
    wavebands : torch.Tensor
        The ``N`` sampled wavelengths [nm], i.e. the grid the network expects.
    input_size : int
        Number of wavelength samples ``N``.
    """

    def __init__(self, model, name=None, description=""):
        self.model = model
        self.name = name
        self.description = description
        fwd = model.forward_model
        if fwd is None:
            raise ValueError("the checkpoint has no forward_model; cannot infer its wavelength setup")
        self.lambda0 = float(fwd.lambda0)
        self.landeG = float(fwd.G)
        self.wing = float(fwd.wing)
        wavebands = fwd.wavebands.detach().cpu().reshape(-1)
        # MEForward prepends the wing point, which is dropped from the output
        self.wavebands = wavebands[1:].clone()
        self.input_size = int(model.layers[0]) // 32 + 4
        self._device = torch.device(model.device)

    # -- introspection ------------------------------------------------------ #
    def __repr__(self):
        return (
            f"<PI2NNInitialGuess {self.name or ''} lambda0={self.lambda0} "
            f"G={self.landeG} N={self.input_size} wavebands={self.wavebands.tolist()}>"
        )

    def describe(self) -> str:
        return (
            f"{self.name or 'PI2NN'}: {self.description}\n"
            f"    lambda0 = {self.lambda0} nm, landeG = {self.landeG}, "
            f"wing = {self.wing} nm\n"
            f"    {self.input_size} wavelength samples: "
            f"{[round(float(w), 6) for w in self.wavebands]}"
        )

    # -- configuration check ------------------------------------------------ #
    def check_inversion(self, inversion, tol_lambda=1e-4, tol_waveband=1e-4, strict=True):
        """Verify that ``inversion`` reproduces the setup the network was trained on.

        The network is a function of the sampled Stokes vector only, so a
        different ``lambda0`` / ``landeG`` / wavelength grid silently means a
        different physical configuration.  ``tol_*`` are tolerances in nm
        (1e-4 nm = 1 mA).
        """
        problems = []
        n_obs = int(inversion.wavebands.numel()) - 1
        if n_obs != self.input_size:
            problems.append(
                f"wavelength samples: inversion has {n_obs}, model expects {self.input_size}"
            )
        if abs(float(inversion.lambda0) - self.lambda0) > tol_lambda:
            problems.append(
                f"lambda0: inversion has {float(inversion.lambda0)}, model expects {self.lambda0}"
            )
        if abs(float(inversion.G) - self.landeG) > 1e-6:
            problems.append(
                f"landeG: inversion has {float(inversion.G)}, model expects {self.landeG}"
            )
        if n_obs == self.input_size:
            dev = (inversion.wavebands[1:].detach().cpu().double()
                   - self.wavebands.double()).abs().max().item()
            if dev > tol_waveband:
                problems.append(
                    f"wavelength grid differs by up to {dev * 1e3:.4g} mA "
                    f"(inversion {[round(float(w), 6) for w in inversion.wavebands[1:]]}, "
                    f"model {[round(float(w), 6) for w in self.wavebands]})"
                )
        if problems:
            msg = (
                f"initial_guess='{self.name or 'PI2NN'}' is not consistent with this inversion:\n  - "
                + "\n  - ".join(problems)
                + "\n  Build the inversion the way the network was trained, e.g.\n"
                f"    MEInversion(wavebands=torch.tensor({[round(float(w), 6) for w in self.wavebands]}), "
                f"landeG={self.landeG}, lambda0={self.lambda0}, wing={self.wing})\n"
                "  or pass strict=False to use the network anyway."
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    # -- inference ---------------------------------------------------------- #
    def _to_device(self, device):
        device = torch.device(device)
        if self._device != device:
            self.model.net.to(device)
            self.model.device = device
            self._device = device
        return device

    def predict_physical(self, iquv_obs: torch.Tensor) -> torch.Tensor:
        """Physical ME parameters ``(B, 8)`` predicted from ``iquv_obs[B, 4, N]``.

        The Stokes vector must be continuum normalised (``I/Ic``, ``Q/Ic``,
        ``U/Ic``, ``V/Ic``), i.e. the same quantity that
        :meth:`CuSP.me_forward.MEForward.return_IQUV` returns.
        """
        if iquv_obs.dim() != 3 or iquv_obs.size(1) != 4:
            raise ValueError(
                f"iquv_obs must have shape (B, 4, N) with channels (I, Q, U, V); "
                f"got {tuple(iquv_obs.shape)}"
            )
        if iquv_obs.size(2) != self.input_size:
            raise ValueError(
                f"the model expects {self.input_size} wavelength samples, "
                f"but iquv_obs has {iquv_obs.size(2)}"
            )
        device = self._to_device(iquv_obs.device)
        x = iquv_obs.detach().to(device=device, dtype=torch.float32)
        with torch.no_grad():
            params = self.model.forward(x)
        return params

    def predict_normalized(self, iquv_obs: torch.Tensor, inversion) -> torch.Tensor:
        """Normalised initial guess ``(B, 8)`` in ``inversion``'s parameter box.

        The network predicts *physical* parameters in its own normalisation
        (``a_damp`` and ``Bmag`` are scaled linearly there, logarithmically in
        :class:`MEInversion`), so the conversion goes through the physical
        values via :meth:`MEInversion.normalizing_parameter`.
        """
        physical = self.predict_physical(iquv_obs)
        xn = inversion.normalizing_parameter(physical.to(device=inversion.wavebands.device))
        xn = torch.nan_to_num(xn, nan=0.5, posinf=1.0, neginf=0.0)
        return xn.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def get_initial_guess_model(name="sdo_hmi", device=None) -> PI2NNInitialGuess:
    """Load (and cache) a registered model, or a checkpoint path."""
    key = str(name)
    wrapper = _MODEL_CACHE.get(key)
    if wrapper is None:
        spec = _REGISTRY.get(key)
        wrapper = PI2NNInitialGuess(
            load_pi2nn_model(key, map_location="cpu", device=device),
            name=key,
            description=spec.description if spec is not None else key,
        )
        _MODEL_CACHE[key] = wrapper
    return wrapper


def initial_guess_from_spectrum(inversion, iquv_obs: torch.Tensor, model="sdo_hmi",
                                strict=True, **kwargs) -> torch.Tensor:
    """Normalised ``(B, 8)`` initial guess for ``inversion`` from ``iquv_obs``.

    Parameters
    ----------
    inversion : MEInversion
        The inversion whose parameter box and wavelength setup are used.
    iquv_obs : torch.Tensor
        Continuum-normalised Stokes vector, shape ``(B, 4, N)``.
    model : str or PI2NNInitialGuess
        Registered model name (``'sdo_hmi'``) or an already loaded wrapper.
    strict : bool
        Raise if the inversion setup differs from the network's training setup
        (default); otherwise only warn.
    """
    wrapper = model if isinstance(model, PI2NNInitialGuess) else get_initial_guess_model(
        model, device=iquv_obs.device)
    wrapper.check_inversion(inversion, strict=strict, **kwargs)
    xn = wrapper.predict_normalized(iquv_obs, inversion)
    return xn.to(device=iquv_obs.device, dtype=torch.float32)

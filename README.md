# CuSP — Cuda-supported SpectroPolarimetry

**English** | [中文](README.zh-CN.md)

> **Paper.** `CuSP` accompanies *A synergistic spectropolarimetric inversion via
> gradient-bias annealing and physics-informed neural networks*, by
> G. Y. Chen, Y. Guo, C. J. Díaz Baso, Q. Hao & M. D. Ding,
> **A&A 710, A227 (2026)** — DOI
> [10.1051/0004-6361/202659270](https://doi.org/10.1051/0004-6361/202659270)
> (open access, CC BY 4.0).  Please cite it if you use this code; a ready-to-paste
> BibTeX entry is in [section 11](#11-citation).

`CuSP` is a PyTorch toolkit for the **Milne–Eddington (ME) inversion of solar
spectropolarimetric data**.  It provides a fully differentiable forward model of
the Stokes vector `IQUV`, batched simulated-annealing inversion on GPU/CPU, fast
Voigt line profiles, and a physics-informed neural network (`PI2NN`) that can be
used to hand the inversion a good initial guess instead of starting from random
noise.

```python
import torch
from CuSP import MEInversion

inv = MEInversion(torch.tensor(wavebands), landeG=2.5, lambda0=630.25, wing=wing)
params = inv(iquv_obs)                                    # random start
params = inv(iquv_obs, initial_guess='sdo_hmi')           # PI2NN initial guess
```

---

## Table of contents

1. [Features](#1-features)
2. [Installation](#2-installation)
3. [Repository layout](#3-repository-layout)
4. [The eight ME parameters](#4-the-eight-me-parameters)
5. [Quick start](#5-quick-start)
6. [Initial guesses](#6-initial-guesses)
7. [Line profiles](#7-line-profiles)
8. [Module reference](#8-module-reference)
9. [Notes, caveats and known issues](#9-notes-caveats-and-known-issues)
10. [Reproducing the validation](#10-reproducing-the-validation)
11. [Citation](#11-citation)

---

## 1. Features

| Feature | Module | Notes |
|---|---|---|
| ME forward synthesis of Stokes `IQUV` | `me_forward.py` | analytic Unno–Rachkovsky solution, fully vectorised, differentiable, device-agnostic |
| ME inversion (batched simulated annealing) | `me_inversion.py` + `annealing.py` | `method='gsa'` — generalized simulated annealing with a gradient-biased step selection, i.e. the **GBA** algorithm of the paper (recommended) — or `'csa'` (conjugate SA); GPU-supported |
| Fast Voigt / Faraday–Voigt profiles | `voigt.py` | 7/7 complex rational approximation of the Faddeeva function — no numerical quadrature, ~13× faster than the original trapezoidal implementation |
| Learned initial guesses | `initial_guess.py` | shipped `PI2NN` network for **SDO/HMI** Fe I 6173 Å spectra: `initial_guess='sdo_hmi'` |
| Physics-informed neural network inversion | `PI2NN.py` | `InversionNet` (conv + attention + residual FC) with a physics loss, plus a training loop |
| Example notebooks | `src/CuSP/examples/` | `forward.ipynb`, `inversion.ipynb` |

## 2. Installation

```bash
git clone git@git.nju.edu.cn:gychen/CuSP.git
cd CuSP
pip install -e .
```

Requirements (see `setup.py`): `torch>=1.13.1`, `numpy>=1.24.3`,
`matplotlib>=3.7.2`, `scipy>=1.11.3`.

Without installing, add the source directory to the path:

```python
import sys; sys.path.insert(0, "CuSP/src")
```

All computation runs on CPU; a CUDA device is used automatically when available
(`MEInversion(..., device='cuda')`, or simply keep the tensors on the GPU).

## 3. Repository layout

```
CuSP/
├── setup.py
└── src/CuSP/
    ├── __init__.py          # public API: MEForward, MEInversion, PI2NN helpers
    ├── me_forward.py        # MEForward      : Stokes IQUV forward synthesis
    ├── me_inversion.py      # MEInversion    : batched ME inversion (+ CudaAnnealing)
    ├── annealing.py         # GSA / DualAnnealing building blocks
    ├── voigt.py             # fast Voigt & Faraday–Voigt profiles
    ├── initial_guess.py     # PI2NN initial guesses (load / register / evaluate)
    ├── PI2NN.py             # physics-informed neural network inversion
    ├── data/
    │   └── pi2nn_sdo_hmi.pkl  # trained PI2NN weights for SDO/HMI
    └── examples/
        ├── forward.ipynb    # forward-model tutorial
        └── inversion.ipynb  # inversion tutorial
```

## 4. The eight ME parameters

`MEInversion` works in a normalised 8-D box `[0,1]^8`; the physical parameters
are recovered with `inv.denormalizing_parameter(x)`.  The order is fixed and
shared by the forward model, the inversion and `PI2NN`:

| # | Parameter | Meaning | Unit | Range | Normalised |
|---|---|---|---|---|---|
| 1 | `Dlambda_D` | Doppler width (thermal + microturbulence) | nm (`1e-4 … 5e-2` nm = `0.1 … 50` mÅ) | `v_D_range = [1, 500]` × `1e-4` nm | log |
| 2 | `v_los` | line-of-sight velocity | m/s | `-7000 … 7000` | linear |
| 3 | `eta_0` | ratio of line to continuum absorption | — | `1 … 1000` | log |
| 4 | `S10` | source-function ratio `S1/S0` | — | `0.1 … 10` | log |
| 5 | `a_damp` | Lorentz damping constant `a` of the Voigt profile | — | `0.4 … 0.6` | log |
| 6 | `Bmag` | magnetic field strength | G | `5 … 5000` | log |
| 7 | `theta` | field inclination | rad | `0 … π` | linear |
| 8 | `phi` | field azimuth | rad | `0 … π` | linear |

Ranges follow Centeno et al. (2014), [doi:10.1007/s11207-014-0497-7](https://doi.org/10.1007/s11207-014-0497-7).
They are class attributes of `MEInversion` (`v_D_range`, `v_los_range`, …) and can
be overridden in a subclass.

> `PI2NN` uses the same physical ranges but scales `a_damp` and `Bmag`
> **linearly** instead of logarithmically.  `initial_guess.py` converts between
> the two conventions through the physical values, so mixing them up is not an
> issue as long as you go through the provided API.

## 5. Quick start

### 5.1 Forward synthesis

`MEForward` turns ME parameters into a continuum-normalised Stokes vector.
**Both the observed wavebands and a `wing` reference wavelength must be given**;
the wing point is prepended internally and used to normalise the spectrum, so
the returned profiles correspond exactly to the `wavebands` you passed in.

```python
import numpy as np
import torch
from CuSP import MEForward

lambda0, dlambda, nlambda = 630.25, 4e-4, 100          # nm
ll   = lambda0 - 0.5 * dlambda * nlambda + np.arange(nlambda) * dlambda
wing = ll.min()                                        # continuum reference

forward = MEForward(torch.tensor(ll).float(), landeG=2.5, lambda0=lambda0, wing=wing)

B = 8                                                  # number of pixels
p = lambda lo, hi, log=True: (                      # uniform / log-uniform sampler
    torch.exp(torch.rand(B, 1) * (np.log(hi) - np.log(lo)) + np.log(lo))
    if log else torch.rand(B, 1) * (hi - lo) + lo)

Dlambda_D, v_los, eta_0, S10 = p(1e-4, 5e-2), p(-7e3, 7e3, False), p(1, 1e3), p(0.1, 10)
a_damp, Bmag, theta, phi = p(0.4, 0.6, False), p(5, 5e3), p(0, np.pi, False), p(0, np.pi, False)

I, Q, U, V = forward(Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi)   # (B, N) each
iquv = torch.stack([I, Q, U, V], dim=1)                                        # (B, 4, N)
```

Every input must broadcast to the same shape (typically `(B, 1)`), and the call
returns a single `(4, B, N)` tensor that unpacks into `I, Q, U, V`.

### 5.2 ME inversion

The example below inverts a *sampled* spectrum (`lm`, a 7-point instrument-like
sampling); the annealing prints its progress as it cools.

```python
from CuSP import MEInversion

inv = MEInversion(torch.tensor(lm).float(), landeG=2.5, lambda0=630.25, wing=wing)

# (a) generalized / gradient-bias annealing from a random start (the paper's GBA)
params = inv(iquv_obs, maxiter=200, initial_temp=5230.)

# (b) conjugate simulated annealing (older method)
params = inv(iquv_obs, method='csa', max_iter=1000)

# (c) start from a PI2NN prediction -- the shipped network is tied to the SDO/HMI
#     sampling, so it needs its own inversion object: see section 6.
#     params = inv_hmi(iquv_obs_hmi, initial_guess='sdo_hmi', maxiter=100)
```

`iquv_obs` has shape `(B, 4, N)` (channels ordered `I, Q, U, V`, continuum
normalised, i.e. exactly what `MEForward` returns).  The returned `params` is the
physical `(B, 8)` parameter array; the normalised start/end points and the merit
values are stored alongside it:

```python
inv.ivs_results.keys()   # dict_keys(['x0', 'e0', 'x', 'e'])
#   x0 / e0 : denormalised initial guess and its chi2/F
#   x  / e  : inverted parameters and their chi2/F
```

Useful keyword arguments (pass through `inv(...)`):

| Keyword | Default | Meaning |
|---|---|---|
| `method` | `'gsa'` | `'gsa'` or `'csa'` |
| `initial_guess` | `None` | see section 6 |
| `x_guess` | `None` | explicit normalised `(B,8)` start; takes precedence over `initial_guess` |
| `maxiter` / `max_iter` | `1000` | annealing iterations per temperature |
| `initial_temp` | `5230.` | starting temperature |
| `visit`, `accept`, `no_local_search` | `2.62`, `-5.0`, `False` | `gsa` control parameters |
| `adam` | `{}`, i.e. `nepoch=1000` | local Adam refinement run after the annealing; pass `adam=dict(nepoch=0)` to skip it, or `dict(nepoch=2000, learning_rate=1e-3)` to tune it |
| `max_batches` | `1e10` | split large batches into chunks of this size |
| `device` | `iquv_obs.device` | computation device |

### 5.3 Plotting the result

```python
import matplotlib.pyplot as plt

inv.plot_annealing_hist()                                     # cooling history
plt.figure(figsize=(8, 8))
inv.plot_inversion_results(choice_index=0)                    # observation vs inversion
inv.plot_inversion_results(choice_index=0, params_obs=truth)  # + the true model, if known
```

### 5.4 Neural-network inversion (`PI2NN`)

`PI2NN` learns the inverse mapping `Stokes -> ME parameters` and is trained with a
combined data loss and physics loss (the physics term re-synthesises the spectrum
and compares it with the observation).

```python
import torch
from CuSP.PI2NN import PI2NN          # note: the class lives inside the PI2NN module

net = PI2NN(hidden_layers=[64] * 2, input_size=6, output_size=8, device='cpu',
            use_residual=False,
            forward_params=dict(wavebands=lm.tolist(), landeG=2.5,
                                lambda0=630.25, wing=float(lm.min())))

net.train(max_iter=1000, training_set=dict(total_size=int(1e6)),
          batch_size=20000, print_interval=10, save_interval=500,
          save_path='./PI2NN_models/', save_name='model',
          do_physics_informed=True, weight_physcis=1.0, dense_spectrum=False)

params = net(iquv_obs)                # physical parameters, shape (B, 8)
```

* `input_size` is the **number of wavelength samples**; the ConvNet flattens to
  `(input_size - 4) * 32` features, so the input tensor is `(B, 4, input_size)`.
* If `forward_params` is given, training data are generated on the fly when
  `training_set` provides no `inputs`.
* ⚠️ `PI2NN` **overrides `nn.Module.train`** with its training loop, and
  `nn.Module.eval()` is implemented as `self.train(False)` — never call
  `net.eval()` on a `PI2NN` instance (it would start training).  Use
  `net.training = False; net.net.eval()`.

## 6. Initial guesses

Starting the annealing from a good guess is what makes large HMI-style data sets
tractable.  `MEInversion.__call__` accepts

| `initial_guess=` | Behaviour |
|---|---|
| `None` (default) | uniform random start in `[0,1]^8` |
| `'sdo_hmi'` | evaluate the shipped PI2NN network on `iquv_obs` |
| other `str` | a name registered with `register_initial_guess`, or a path to a `.pkl` saved by `torch.save(PI2NN_instance, path)` |
| `torch.Tensor` | explicit normalised guess, shape `(B, 8)` or `(8,)` |
| callable | `f(iquv_obs, inversion=self, device=...)` returning physical or normalised `(B, 8)` parameters |

```python
from CuSP import MEInversion, get_initial_guess_model, list_initial_guess_models

print(list_initial_guess_models())          # {'sdo_hmi': 'PI2NN trained on ...'}

ig = get_initial_guess_model('sdo_hmi')     # wrapper, with the model's own setup
print(ig.describe())

inv = MEInversion(ig.wavebands, landeG=ig.landeG, lambda0=ig.lambda0, wing=ig.wing)
x0   = inv.make_initial_guess(iquv_obs, initial_guess='sdo_hmi')   # normalised (B,8)
phys = ig.predict_physical(iquv_obs)                               # physical (B,8)
params = inv(iquv_obs, initial_guess='sdo_hmi')                    # or go straight ahead
```

### The shipped `sdo_hmi` model

* Trained for **SDO/HMI** spectra of Fe I 6173.34 Å: 6 wavelength samples at
  617.317139, 617.324036, 617.330872, 617.337769, 617.344666, 617.351501 nm
  (HMI's ±34.4 / ±103.2 / ±172 mÅ tuning points), `landeG = 2.5`,
  `lambda0 = 617.33352 nm`, 2000 training epochs.
* Because a network is only valid for the sampling it was trained on,
  `initial_guess_from_spectrum` **checks the inversion configuration** (number of
  samples, `lambda0`, `landeG`, wavelength grid; tolerance `1e-4` nm) and raises a
  `ValueError` with the correct `MEInversion(...)` call if they disagree.  Pass
  `strict=False` to downgrade this to a warning.
* The input spectrum must be **continuum normalised** (`I/Ic, Q/Ic, U/Ic, V/Ic`).
* Measured performance (8-point forward model on CPU, synthetic HMI-like
  spectra): ~5300 spectra/s; the network's own spectra are within ≈0.22σ of the
  truth, so it is an excellent *starting point* (median `chi2/F` 6e-2 vs 1.9 for
  a random start) but not a final answer — after 40 annealing iterations the
  inversion reaches `1.6e-4` vs `5.6e-3` (≈36× better).
* ⚠️ **Wavelength convention to be aware of**: `lambda0 = 6173.3352 Å` is the
  Fe I rest wavelength while the six filter points are centred on the observed
  disk-centre line position 6173.3433 Å (a difference of 8.1 mÅ ≈ 393 m/s).
  The network's `v_los` is therefore measured relative to 6173.3352 Å.  Check
  that this matches the convention of your HMI reduction before interpreting
  absolute velocities.

Registering your own network is one line:

```python
from CuSP import register_initial_guess
register_initial_guess('my_line', '/path/to/pi2nn_mine.pkl', 'PI2NN for Fe I 6302.5')
params = inv(iquv_obs, initial_guess='my_line')
```

## 7. Line profiles

The Voigt `φ(u,a)` and Faraday–Voigt `ψ(u,a)` profiles are the expensive inner
loop of the ME equations.  `voigt.py` evaluates them through a **7/7 complex
rational approximation of the Faddeeva function**, with no numerical integration:

```python
from CuSP.voigt import VoigtProfile, VoigtFaradayProfile, voigt_profiles

phi = VoigtProfile(u, a)              # (B, Nw) — u (B, Nw), a (B, 1)
psi = VoigtFaradayProfile(u, a)       # (B, Nw)
phi, psi = voigt_profiles(u, a)       # both at once

from CuSP.voigt import VoigtProfileQuadrature    # original trapezoidal version
phi_slow = VoigtProfileQuadrature(u, a, ynodes=1000, lim=10.0)
```

Conventions (unchanged from the original code, so nothing downstream moved):

```
phi(u, a) = Re w(u + i a) / sqrt(pi)      with   int phi du = 1
psi(u, a) = Im w(u + i a) / sqrt(pi)      (Hilbert partner of phi)
```

`Re w` is even and `Im w` is odd in `u`, and the rational form is accurate for
both signs of `u` and for all `a ≥ 0`.  Measured absolute error against
`scipy.special.wofz` over `|u| ≤ 200`:

| `a` | rational (default) | trapezoidal reference |
|---|---|---|
| 0 | 2.5e-6 | **returns exactly 0 (and NaN in places)** |
| 1e-4 | 2.5e-6 | 5.6e-1 |
| 0.01 | 2.4e-6 | 4.7e-2 |
| 0.1 | 1.4e-6 | 1.0e-5 |
| 0.5 | 2.2e-7 | 7.9e-6 (float32) / 6e-17 (float64) |
| 1.0 | 3.6e-8 | 1e-5 (float32) |

The `ynodes` / `lim` arguments of the old quadrature are accepted and ignored.
`|z| ≳ 1.4e5` overflows the polynomial in float32 (CuSP's operating range is
`|u| ≲ 600`, i.e. ~200× inside the limit).

## 8. Module reference

**`CuSP` (top level exports)**

```python
from CuSP import (MEForward, MEInversion, PI2NNInitialGuess, get_initial_guess_model,
                  initial_guess_from_spectrum, list_initial_guess_models,
                  load_pi2nn_model, register_initial_guess)
```

**`me_forward.MEForward(wavebands, landeG=2.5, lambda0=630.25, wing=None)`**

| Method | Returns |
|---|---|
| `__call__(Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi)` | `(4, B, N)` Stokes `I, Q, U, V` |
| `return_IQUV(...)` | tuple `(I, Q, U, V)`, each `(B, N)` |
| `return_profile(Dlambda_D, u_los, u_B, a_damp)` | `(φ₀, φ_B, φ_R, ψ₀, ψ_B, ψ_R)` |
| `return_eta_rho(eta_0, theta, phi, φ…, ψ…)` | propagation-matrix elements `η_{I,Q,U,V}`, `ρ_{Q,U,V}` |

**`me_inversion.MEInversion(wavebands, landeG=2.5, lambda0=630.25, wing=None)`**
(inherits `MEForward`; overrides `__call__` as the inversion entry point, so use
`inv.synthesize(params)` / `inv.return_IQUV(...)` for forward synthesis)

`__call__`, `make_initial_guess`, `synthesize`, `merit_function`,
`normalizing_parameter`, `denormalizing_parameter`, `plot_inversion_results`,
`plot_annealing_hist`, and the results dict `ivs_results`.

**`annealing.py`** — `DualAnnealing(func, bounds, x0, maxiter, adam, initial_temp,
visit, accept, no_local_search)`, plus `GSA`, `VisitDistribution`, `EnergyState`,
`AdamLocalSearch`, `BatchAdam`, `AnnealingResult`.
**`me_inversion.CudaAnnealing`** — the `csa` driver.

**`initial_guess.py`** — `load_pi2nn_model`, `PI2NNInitialGuess`
(`predict_physical`, `predict_normalized`, `check_inversion`, `describe`),
`get_initial_guess_model`, `initial_guess_from_spectrum`,
`list_initial_guess_models`, `register_initial_guess`.

**`voigt.py`** — `VoigtProfile`, `VoigtFaradayProfile`, `voigt_profiles`,
`faddeeva_rational`, `VoigtProfileQuadrature`, `VoigtFaradayProfileQuadrature`,
`VOIGT_RATIONAL_A/B`.

**`PI2NN.py`** — `PI2NN` (training loop + `forward`), `InversionNet`, `FCN`,
`ResidualBlock`, `ConvResidualBlock`, `ChannelAttention`, `SelfAttention`.

## 9. Notes, caveats and known issues

* **Normalisation.** Wavebands are in nm and the Stokes vector is normalised at
  the `wing` wavelength; feed the inversion continuum-normalised data
  (`I/Ic … V/Ic`).
* **Wavelength reference.** `MEForward` computes `u = (λ − lambda0)/Dlambda_D`, so
  `lambda0` must be the line centre you want `v_los` measured against.  The
  shipped `sdo_hmi` model is deliberately checked against this (section 6).
* **dtype.** float32 is the working precision (the annealer casts internally);
  float64 is supported and gives a more accurate quadrature reference.
* **`PI2NN` and `nn.Module`.** `PI2NN.train` is the training loop, so `.eval()`
  must not be used on a `PI2NN` instance (see section 5.4).  Loading the shipped
  checkpoint does not hit this, but your own code might.
* **`method='csa'`** used to crash with `UnboundLocalError: E_init`; this is
  fixed (`CudaAnnealing._annealing`).  `'gsa'` remains the recommended method.
* **`initial_guess.py` loading.** The shipped checkpoint was saved from a script
  that imported `PI2NN` as a *top-level* module, so `torch.load` alone cannot
  unpickle it; `load_pi2nn_model` remaps those class references to
  `CuSP.PI2NN`.  `torch>=2.6` also requires `weights_only=False`, which the
  loader sets when supported.

## 10. Reproducing the validation

The numbers quoted above come from scripts maintained next to this repository
(`../analysis/` in the author's workspace):

| Script | What it checks |
|---|---|
| `verify_voigt_port.py` | Voigt profiles against `scipy.special.wofz` and against the pre-port code (loaded from `git show HEAD:`), shapes/dtypes/autograd, and the speed-up |
| `probe_rational_domain.py` | validity domain of the rational approximation (signs of `u`, `a → 0`, large `|u|`, float32) |
| `inspect_pi2nn_pkl.py` | static inspection of a checkpoint's class references |
| `probe_pi2nn_model.py` | the stored `bounds` / `norm_scale` / `forward_model` of a checkpoint |
| `validate_pi2nn_initial_guess.py` | end-to-end check of `initial_guess='sdo_hmi'`: network accuracy, annealing improvement, configuration guard |
| `smoke_pi2nn.py` | `PI2NN` training smoke test (both `dense_spectrum` paths) |
| `demo_initial_guess_sdo_hmi.py` | the `initial_guess='sdo_hmi'` usage shown above |
| `verify_readme_snippets.py` | runs every code snippet of this README verbatim, so the documented API cannot drift from the implementation |

## 11. Citation

If you use `CuSP` in a publication, please cite the accompanying paper:

> G. Y. Chen, Y. Guo, C. J. Díaz Baso, Q. Hao & M. D. Ding,
> *A synergistic spectropolarimetric inversion via gradient-bias annealing and
> physics-informed neural networks*,
> Astronomy & Astrophysics **710**, A227 (2026).
> DOI: [10.1051/0004-6361/202659270](https://doi.org/10.1051/0004-6361/202659270)
> (open access, CC BY 4.0)

```bibtex
@ARTICLE{Chen2026synergistic,
  author  = {{Chen}, G.~Y. and {Guo}, Y. and {D{\'i}az Baso}, C.~J. and
             {Hao}, Q. and {Ding}, M.~D.},
  title   = "{A synergistic spectropolarimetric inversion via gradient-bias
             annealing and physics-informed neural networks}",
  journal = {A\&A},
  year    = {2026},
  volume  = {710},
  pages   = {A227},
  doi     = {10.1051/0004-6361/202659270}
}
```

---

**Author:** Chen Guoyin · gychen@smail.nju.edu.cn · version 0.1.0

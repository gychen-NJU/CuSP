import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CuSP import MEForward, MEInversion, get_initial_guess_model

HERE = Path(__file__).resolve().parent

PARAM_NAMES = ["Dlambda_D", "v_los", "eta_0", "S10", "a_damp", "Bmag", "theta", "phi"]
PARAM_UNITS = ["nm", "m/s", "-", "-", "-", "G", "rad", "rad"]
STOKES = [r"$I/I_c$", r"$Q/I_c$", r"$U/I_c$", r"$V/I_c$"]
TRUTH = np.array([4.0e-3, 393.0, 30.0, 1.0, 0.5, 1500.0, np.pi / 4, np.pi / 6])
CENTER_A = 6173.343200683594


def print_table(header, rows):
    widths = [max(len(str(line[i])) for line in [header] + rows) for i in range(len(header))]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def synthesize(forward, params, dtype=torch.float32):
    params = torch.tensor(np.asarray(params, dtype=np.float64).reshape(1, 8), dtype=dtype)
    with torch.no_grad():
        out = forward(*[params[:, k:k + 1] for k in range(8)])
    return out[:, 0, :].numpy()


def plot_comparison(path, title, wavelengths, dense_wavelengths, target_points, target_dense,
                    inverted_points, inverted_dense):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for k, ax in enumerate(axes.ravel()):
        ax.plot(dense_wavelengths * 10.0, target_dense[k], "-", color="k", lw=2.4, alpha=0.30,
                label="target (ME model)")
        ax.plot(wavelengths * 10.0, target_points[k], "o", ms=7, mfc="none", mec="k", mew=1.2,
                label="target at the 6 HMI wavelengths")
        ax.plot(dense_wavelengths * 10.0, inverted_dense[k], "--", color="C3", lw=1.3,
                label="inverted (dense)")
        ax.plot(wavelengths * 10.0, inverted_points[k], "x", ms=7, color="C3", mew=1.5,
                label="inverted at the 6 HMI wavelengths")
        ax.set_xlabel(r"$\lambda$  [$\mathrm{\AA}$]")
        ax.set_ylabel(STOKES[k])
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="x", useOffset=False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


ig = get_initial_guess_model("sdo_hmi")
WAVEBANDS = ig.wavebands.clone()
LAMBDA0 = ig.lambda0
LANDE_G = ig.landeG
WING = ig.wing
NATIVE = ig.wavebands.numpy()
DENSE = np.linspace(CENTER_A / 10.0 - 0.018, CENTER_A / 10.0 + 0.018, 400)

inv = MEInversion(WAVEBANDS.clone(), landeG=LANDE_G, lambda0=LAMBDA0, wing=WING)
forward_hmi = MEForward(WAVEBANDS.clone(), landeG=LANDE_G, lambda0=LAMBDA0, wing=WING)
forward_dense = MEForward(torch.tensor(DENSE, dtype=torch.float32), landeG=LANDE_G,
                          lambda0=LAMBDA0, wing=WING)

torch.manual_seed(0)
np.random.seed(0)

obs = inv.synthesize(torch.tensor(TRUTH, dtype=torch.float32).reshape(1, 8))
x_guess = inv.make_initial_guess(obs, initial_guess=None)
merit_start = float(np.asarray(inv.merit_function(
    obs, inv.synthesize(inv.denormalizing_parameter(x_guess)))).reshape(-1)[0])

params_cmaes = inv(obs, method="cmaes", max_iter=500, initial_guess=x_guess, isPrint=False)
merit_cmaes = float(np.asarray(inv.ivs_results["e"]).reshape(-1)[0])

x_refine = inv.normalizing_parameter(torch.as_tensor(params_cmaes))
params_lm = inv(obs, method="lm", max_iter=60, initial_guess=x_refine, isPrint=False)
merit_lm = float(np.asarray(inv.ivs_results["e"]).reshape(-1)[0])

params_cmaes = params_cmaes.detach().numpy()[0]
params_lm = params_lm.detach().numpy()[0]

print()
print_table(
    ["parameter", "unit", "truth", "cmaes", "cmaes+lm", "rel.err(cmaes+lm)"],
    [[name, unit, f"{truth:.6g}", f"{cma:.6g}", f"{lm:.6g}", f"{abs(lm - truth) / abs(truth):.3e}"]
     for name, unit, truth, cma, lm in
     zip(PARAM_NAMES, PARAM_UNITS, TRUTH, params_cmaes, params_lm)])
print()
print(f"chi2/dof  random start  : {merit_start:.4e}")
print(f"chi2/dof  after cmaes   : {merit_cmaes:.4e}")
print(f"chi2/dof  after lm      : {merit_lm:.4e}")

target_points = synthesize(forward_hmi, TRUTH)
inverted_points = synthesize(forward_hmi, params_lm)
target_dense = synthesize(forward_dense, TRUTH)
inverted_dense = synthesize(forward_dense, params_lm)

plot_path = HERE / "inversion_random_cmaes_lm.png"
plot_comparison(
    plot_path,
    "Random initial guess -> CMA-ES -> Levenberg-Marquardt  (SDO/HMI Fe I 6173 A)",
    NATIVE, DENSE, target_points, target_dense, inverted_points, inverted_dense)
print()
print(f"figure saved to {plot_path}")

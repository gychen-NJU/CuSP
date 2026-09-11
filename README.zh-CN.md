# CuSP — Cuda-supported SpectroPolarimetry

[English](README.md) | **中文**

> **技术文章。** `CuSP` 对应论文 *A synergistic spectropolarimetric inversion via
> gradient-bias annealing and physics-informed neural networks*，作者
> G. Y. Chen、Y. Guo、C. J. Díaz Baso、Q. Hao、M. D. Ding，
> **A&A 710, A227 (2026)**，DOI
> [10.1051/0004-6361/202659270](https://doi.org/10.1051/0004-6361/202659270)
> （开放获取，CC BY 4.0）。使用本代码时请引用该文；可直接复制的 BibTeX 见
> [第 11 节](#11-引用)。

`CuSP` 是一个基于 PyTorch 的**太阳偏振光谱 Milne–Eddington（ME）反演**工具包。
它提供完全可微的 Stokes `IQUV` 前向模型、GPU/CPU 上的批量模拟退火反演、快速的
Voigt 线型计算，以及一个物理信息神经网络（`PI2NN`）——可以据此给反演一个良好
的初猜，而不是从随机噪声出发。

```python
import torch
from CuSP import MEInversion

inv = MEInversion(torch.tensor(wavebands), landeG=2.5, lambda0=630.25, wing=wing)
params = inv(iquv_obs)                                    # 随机初猜
params = inv(iquv_obs, initial_guess='sdo_hmi')           # 用 PI2NN 给初猜
```

---

## 目录

1. [功能概览](#1-功能概览)
2. [安装](#2-安装)
3. [仓库结构](#3-仓库结构)
4. [ME 的八个参数](#4-me-的八个参数)
5. [快速上手](#5-快速上手)
6. [初猜](#6-初猜)
7. [线型函数](#7-线型函数)
8. [模块速查](#8-模块速查)
9. [注意事项与已知问题](#9-注意事项与已知问题)
10. [复现验证](#10-复现验证)
11. [引用](#11-引用)

---

## 1. 功能概览

| 功能 | 模块 | 说明 |
|---|---|---|
| ME 前向合成 Stokes `IQUV` | `me_forward.py` | Unno–Rachkovsky 解析解，全向量化、可微、与设备无关 |
| ME 反演（批量模拟退火） | `me_inversion.py` + `annealing.py` | `method='gsa'`——带梯度偏置步长选择的广义模拟退火，即论文的 **GBA** 算法（推荐）；或 `'csa'`（共轭模拟退火）；支持 GPU |
| 快速 Voigt / Faraday–Voigt 线型 | `voigt.py` | 用 Faddeeva 函数的 7/7 复有理逼近——不做数值积分，比原梯形积分快约 13 倍 |
| 学习型初猜 | `initial_guess.py` | 内置 **SDO/HMI** Fe I 6173 Å 的 `PI2NN` 网络：`initial_guess='sdo_hmi'` |
| 物理信息神经网络反演 | `PI2NN.py` | `InversionNet`（卷积 + 注意力 + 残差全连接）+ 物理损失，含训练循环 |
| 示例 notebook | `src/CuSP/examples/` | `forward.ipynb`、`inversion.ipynb` |

## 2. 安装

```bash
git clone git@git.nju.edu.cn:gychen/CuSP.git
cd CuSP
pip install -e .
```

依赖（见 `setup.py`）：`torch>=1.13.1`、`numpy>=1.24.3`、`matplotlib>=3.7.2`、
`scipy>=1.11.3`。

不安装也可以，把源码目录加进路径即可：

```python
import sys; sys.path.insert(0, "CuSP/src")
```

所有计算都能在 CPU 上跑；有 CUDA 时自动使用（`MEInversion(..., device='cuda')`，
或直接把张量放在 GPU 上）。

## 3. 仓库结构

```
CuSP/
├── setup.py
└── src/CuSP/
    ├── __init__.py          # 对外 API：MEForward、MEInversion、PI2NN 相关工具
    ├── me_forward.py        # MEForward      ：Stokes IQUV 前向合成
    ├── me_inversion.py      # MEInversion    ：批量 ME 反演（含 CudaAnnealing）
    ├── annealing.py         # GSA / DualAnnealing 的构件
    ├── voigt.py             # 快速 Voigt 与 Faraday–Voigt 线型
    ├── initial_guess.py     # PI2NN 初猜（加载 / 注册 / 推理）
    ├── PI2NN.py             # 物理信息神经网络反演
    ├── data/
    │   └── pi2nn_sdo_hmi.pkl  # 面向 SDO/HMI 训练好的 PI2NN 权重
    └── examples/
        ├── forward.ipynb    # 前向模型教程
        └── inversion.ipynb  # 反演教程
```

## 4. ME 的八个参数

`MEInversion` 在归一化的 8 维方盒 `[0,1]^8` 内工作，物理量用
`inv.denormalizing_parameter(x)` 还原。参数顺序固定，前向模型、反演与 `PI2NN`
三者一致：

| # | 参数 | 含义 | 单位 | 范围 | 归一化 |
|---|---|---|---|---|---|
| 1 | `Dlambda_D` | 多普勒宽度（热运动 + 微湍流） | nm（`1e-4 … 5e-2` nm，即 `0.1 … 50` mÅ） | `v_D_range = [1, 500]` × `1e-4` nm | 对数 |
| 2 | `v_los` | 视向速度 | m/s | `-7000 … 7000` | 线性 |
| 3 | `eta_0` | 线吸收与连续吸收之比 | — | `1 … 1000` | 对数 |
| 4 | `S10` | 源函数比 `S1/S0` | — | `0.1 … 10` | 对数 |
| 5 | `a_damp` | Voigt 线型的洛伦兹阻尼常数 `a` | — | `0.4 … 0.6` | 对数 |
| 6 | `Bmag` | 磁场强度 | G | `5 … 5000` | 对数 |
| 7 | `theta` | 磁场倾角 | rad | `0 … π` | 线性 |
| 8 | `phi` | 磁场方位角 | rad | `0 … π` | 线性 |

范围取自 Centeno et al. (2014)，[doi:10.1007/s11207-014-0497-7](https://doi.org/10.1007/s11207-014-0497-7)。
它们是 `MEInversion` 的类属性（`v_D_range`、`v_los_range` …），可在子类里覆盖。

> `PI2NN` 用同样的物理范围，但把 `a_damp` 与 `Bmag` 按**线性**（而非对数）归一。
> `initial_guess.py` 中间经过物理量换算，只要走提供的 API 就不会混。

## 5. 快速上手

### 5.1 前向合成

`MEForward` 把 ME 参数变成连续谱归一化的 Stokes 矢量。**必须同时给出观测波段和
一个 `wing` 参考波长**；wing 点会在内部被插到最前面用于归一化，所以返回的谱
正好对应你传入的 `wavebands`。

```python
import numpy as np
import torch
from CuSP import MEForward

lambda0, dlambda, nlambda = 630.25, 4e-4, 100          # nm
ll   = lambda0 - 0.5 * dlambda * nlambda + np.arange(nlambda) * dlambda
wing = ll.min()                                        # 连续谱参考波长

forward = MEForward(torch.tensor(ll).float(), landeG=2.5, lambda0=lambda0, wing=wing)

B = 8                                                  # 像素数

def sample(lo, hi, log=False):
    """均匀 / 对数均匀采样，返回 (B,1) 张量。"""
    r = torch.rand(B, 1)
    return torch.exp(r * (np.log(hi) - np.log(lo)) + np.log(lo)) if log else r * (hi - lo) + lo

Dlambda_D, v_los, eta_0, S10 = sample(1e-4, 5e-2, True), sample(-7e3, 7e3), sample(1, 1e3, True), sample(0.1, 10, True)
a_damp, Bmag, theta, phi = sample(0.4, 0.6), sample(5, 5e3, True), sample(0, np.pi), sample(0, np.pi)

I, Q, U, V = forward(Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi)   # 各 (B, N)
iquv = torch.stack([I, Q, U, V], dim=1)                                        # (B, 4, N)
```

每个输入都要能广播到同一形状（通常 `(B, 1)`），调用返回一个 `(4, B, N)` 张量，
可直接解包成 `I, Q, U, V`。

### 5.2 ME 反演

下面的例子反演的是*采样后*的光谱（`lm`，7 点，接近仪器采样）；退火过程中会
打印降温进度。

```python
from CuSP import MEInversion

inv = MEInversion(torch.tensor(lm).float(), landeG=2.5, lambda0=630.25, wing=wing)

# (a) 随机初猜 + 广义/梯度偏置模拟退火（即论文的 GBA）
params = inv(iquv_obs, maxiter=200, initial_temp=5230.)

# (b) 共轭模拟退火（较老的方法）
params = inv(iquv_obs, method='csa', max_iter=1000)

# (c) 用 PI2NN 的预测作为退火起点 —— 内置网络绑定的是 SDO/HMI 的采样方式，
#     需要单独构造反演对象：见第 6 节
#     params = inv_hmi(iquv_obs_hmi, initial_guess='sdo_hmi', maxiter=100)
```

`iquv_obs` 形状为 `(B, 4, N)`（通道顺序 `I, Q, U, V`，连续谱归一化，即
`MEForward` 的输出）。返回的 `params` 是物理量 `(B, 8)`；归一化的起点/终点与
判据值同时保存：

```python
inv.ivs_results.keys()   # dict_keys(['x0', 'e0', 'x', 'e'])
#   x0 / e0 : 反归一化的初猜及其 chi2/F
#   x  / e  : 反演结果及其 chi2/F
```

常用关键字参数（直接传给 `inv(...)`）：

| 关键字 | 默认值 | 含义 |
|---|---|---|
| `method` | `'gsa'` | `'gsa'` 或 `'csa'` |
| `initial_guess` | `None` | 见第 6 节 |
| `x_guess` | `None` | 显式的归一化 `(B,8)` 起点，优先于 `initial_guess` |
| `maxiter` / `max_iter` | `1000` | 每个温度下的退火步数 |
| `initial_temp` | `5230.` | 起始温度 |
| `visit`、`accept`、`no_local_search` | `2.62`、`-5.0`、`False` | `gsa` 的控制参数 |
| `adam` | `{}`，即 `nepoch=1000` | 退火之后跑的局部 Adam 精修；传 `adam=dict(nepoch=0)` 可跳过，`dict(nepoch=2000, learning_rate=1e-3)` 可调节 |
| `max_batches` | `1e10` | 大批量按此尺寸分块 |
| `device` | `iquv_obs.device` | 计算设备 |

### 5.3 画图

```python
import matplotlib.pyplot as plt

inv.plot_annealing_hist()                                     # 降温过程
plt.figure(figsize=(8, 8))
inv.plot_inversion_results(choice_index=0)                    # 观测 vs 反演
inv.plot_inversion_results(choice_index=0, params_obs=truth)  # 若知道真值再加一条
```

### 5.4 神经网络反演（`PI2NN`）

`PI2NN` 学习 `Stokes → ME 参数` 的逆映射，训练时同时使用数据损失与物理损失
（物理项重新合成谱并与观测比较）。

```python
import torch
from CuSP.PI2NN import PI2NN          # 注意：类在 PI2NN 模块内部

net = PI2NN(hidden_layers=[64] * 2, input_size=6, output_size=8, device='cpu',
            use_residual=False,
            forward_params=dict(wavebands=lm.tolist(), landeG=2.5,
                                lambda0=630.25, wing=float(lm.min())))

net.train(max_iter=1000, training_set=dict(total_size=int(1e6)),
          batch_size=20000, print_interval=10, save_interval=500,
          save_path='./PI2NN_models/', save_name='model',
          do_physics_informed=True, weight_physcis=1.0, dense_spectrum=False)

params = net(iquv_obs)                # 物理参数，(B, 8)
```

* `input_size` 是**波长采样点数**；卷积网络展平后为 `(input_size - 4) * 32`
  维特征，因此输入张量是 `(B, 4, input_size)`。
* 给了 `forward_params` 且 `training_set` 未提供 `inputs` 时，训练数据会在线生成。
* ⚠️ `PI2NN` **重载了 `nn.Module.train`** 作为训练循环，而 `nn.Module.eval()`
  内部是 `self.train(False)`——所以**不要对 `PI2NN` 实例调用 `net.eval()`**
  （那会真的开始训练）。应写 `net.training = False; net.net.eval()`。

## 6. 初猜

给退火一个好起点，是让 HMI 量级数据集变得可算的关键。`MEInversion.__call__` 接受

| `initial_guess=` | 行为 |
|---|---|
| `None`（默认） | 在 `[0,1]^8` 内均匀随机 |
| `'sdo_hmi'` | 用内置的 PI2NN 网络对 `iquv_obs` 做推理 |
| 其他 `str` | 用 `register_initial_guess` 注册的名字，或一个 `torch.save(PI2NN_instance, path)` 存出的 `.pkl` 路径 |
| `torch.Tensor` | 显式归一化初猜，形状 `(B, 8)` 或 `(8,)` |
| 可调用对象 | `f(iquv_obs, inversion=self, device=...)`，返回物理量或归一化的 `(B, 8)` |

```python
from CuSP import MEInversion, get_initial_guess_model, list_initial_guess_models

print(list_initial_guess_models())          # {'sdo_hmi': 'PI2NN trained on ...'}

ig = get_initial_guess_model('sdo_hmi')     # 包装对象，带模型自身的配置
print(ig.describe())

inv = MEInversion(ig.wavebands, landeG=ig.landeG, lambda0=ig.lambda0, wing=ig.wing)
x0   = inv.make_initial_guess(iquv_obs, initial_guess='sdo_hmi')   # 归一化 (B,8)
phys = ig.predict_physical(iquv_obs)                               # 物理量 (B,8)
params = inv(iquv_obs, initial_guess='sdo_hmi')                    # 一步到位
```

### 内置的 `sdo_hmi` 模型

* 面向 **SDO/HMI** 的 Fe I 6173.34 Å 光谱训练：6 个波长采样点，分别是
  617.317139、617.324036、617.330872、617.337769、617.344666、617.351501 nm
  （HMI 的 ±34.4 / ±103.2 / ±172 mÅ 调谐点），`landeG = 2.5`，
  `lambda0 = 617.33352 nm`，训练 2000 轮。
* 网络只对它训练时的采样方式有效，因此 `initial_guess_from_spectrum` 会**校验
  反演的配置**（采样点数、`lambda0`、`landeG`、波长网格，容差 `1e-4` nm），
  不一致就抛出 `ValueError` 并给出正确的 `MEInversion(...)` 写法；传
  `strict=False` 可降级为警告。
* 输入光谱必须是**连续谱归一化**的（`I/Ic, Q/Ic, U/Ic, V/Ic`）。
* 实测（CPU、合成 HMI 谱）：约 5300 谱/s；网络自身谱与真值相差约 0.22σ，是一个
  很好的**起点**（`chi2/F` 中位数 6e-2，随机初猜为 1.9），但不是最终答案——退火
  40 步后可达 `1.6e-4`（随机初猜为 `5.6e-3`，约优 36 倍）。
* ⚠️ **需要留意的波长约定**：`lambda0 = 6173.3352 Å` 是 Fe I 的静止波长，而 6 个
  滤波点是以观测到的日面中心线心 6173.3433 Å 为中心排布的（差 8.1 mÅ ≈ 393 m/s）。
  因此网络给的 `v_los` 是相对 6173.3352 Å 的。在解释绝对速度前，请先确认这与你的
  HMI 归算约定一致。

注册自己的网络只需一行：

```python
from CuSP import register_initial_guess
register_initial_guess('my_line', '/path/to/pi2nn_mine.pkl', 'PI2NN for Fe I 6302.5')
params = inv(iquv_obs, initial_guess='my_line')
```

## 7. 线型函数

Voigt `φ(u,a)` 与 Faraday–Voigt `ψ(u,a)` 是 ME 方程里最耗时的内层计算。
`voigt.py` 用 **Faddeeva 函数的 7/7 复有理逼近**求值，完全不做数值积分：

```python
from CuSP.voigt import VoigtProfile, VoigtFaradayProfile, voigt_profiles

phi = VoigtProfile(u, a)              # (B, Nw) —— u 为 (B, Nw)，a 为 (B, 1)
psi = VoigtFaradayProfile(u, a)       # (B, Nw)
phi, psi = voigt_profiles(u, a)       # 一次算两个

from CuSP.voigt import VoigtProfileQuadrature    # 原来的梯形积分版本
phi_slow = VoigtProfileQuadrature(u, a, ynodes=1000, lim=10.0)
```

约定与原始代码完全一致（所以下游无需改动）：

```
phi(u, a) = Re w(u + i a) / sqrt(pi)      int phi du = 1
psi(u, a) = Im w(u + i a) / sqrt(pi)      与 phi 构成 Hilbert 对
```

`Re w` 关于 `u` 为偶、`Im w` 为奇，有理逼近对 `u` 的两个符号、以及所有 `a ≥ 0`
都成立。以 `scipy.special.wofz` 为基准、`|u| ≤ 200` 的绝对误差实测：

| `a` | 有理逼近（默认） | 梯形积分参考实现 |
|---|---|---|
| 0 | 2.5e-6 | **恒为 0（个别点还是 NaN）** |
| 1e-4 | 2.5e-6 | 5.6e-1 |
| 0.01 | 2.4e-6 | 4.7e-2 |
| 0.1 | 1.4e-6 | 1.0e-5 |
| 0.5 | 2.2e-7 | 7.9e-6（float32）/ 6e-17（float64） |
| 1.0 | 3.6e-8 | 1e-5（float32） |

旧梯形实现的 `ynodes` / `lim` 参数仍被接受，但会被忽略。float32 下 `|z| ≳ 1.4e5`
时多项式会溢出（CuSP 实际工作范围 `|u| ≲ 600`，还有约 200 倍余量）。

## 8. 模块速查

**`CuSP`（顶层导出）**

```python
from CuSP import (MEForward, MEInversion, PI2NNInitialGuess, get_initial_guess_model,
                  initial_guess_from_spectrum, list_initial_guess_models,
                  load_pi2nn_model, register_initial_guess)
```

**`me_forward.MEForward(wavebands, landeG=2.5, lambda0=630.25, wing=None)`**

| 方法 | 返回 |
|---|---|
| `__call__(Dlambda_D, v_los, eta_0, S10, a_damp, Bmag, theta, phi)` | `(4, B, N)` 的 Stokes `I, Q, U, V` |
| `return_IQUV(...)` | 元组 `(I, Q, U, V)`，各为 `(B, N)` |
| `return_profile(Dlambda_D, u_los, u_B, a_damp)` | `(φ₀, φ_B, φ_R, ψ₀, ψ_B, ψ_R)` |
| `return_eta_rho(eta_0, theta, phi, φ…, ψ…)` | 传播矩阵元 `η_{I,Q,U,V}`、`ρ_{Q,U,V}` |

**`me_inversion.MEInversion(wavebands, landeG=2.5, lambda0=630.25, wing=None)`**
（继承 `MEForward`；并把 `__call__` 覆写为反演入口，因此要合成谱请用
`inv.synthesize(params)` / `inv.return_IQUV(...)`）

`__call__`、`make_initial_guess`、`synthesize`、`merit_function`、
`normalizing_parameter`、`denormalizing_parameter`、`plot_inversion_results`、
`plot_annealing_hist`，以及结果字典 `ivs_results`。

**`annealing.py`** —— `DualAnnealing(func, bounds, x0, maxiter, adam, initial_temp,
visit, accept, no_local_search)`，以及 `GSA`、`VisitDistribution`、`EnergyState`、
`AdamLocalSearch`、`BatchAdam`、`AnnealingResult`。
**`me_inversion.CudaAnnealing`** —— `csa` 的驱动器。

**`initial_guess.py`** —— `load_pi2nn_model`、`PI2NNInitialGuess`
（`predict_physical`、`predict_normalized`、`check_inversion`、`describe`）、
`get_initial_guess_model`、`initial_guess_from_spectrum`、
`list_initial_guess_models`、`register_initial_guess`。

**`voigt.py`** —— `VoigtProfile`、`VoigtFaradayProfile`、`voigt_profiles`、
`faddeeva_rational`、`VoigtProfileQuadrature`、`VoigtFaradayProfileQuadrature`、
`VOIGT_RATIONAL_A/B`。

**`PI2NN.py`** —— `PI2NN`（训练循环 + `forward`）、`InversionNet`、`FCN`、
`ResidualBlock`、`ConvResidualBlock`、`ChannelAttention`、`SelfAttention`。

## 9. 注意事项与已知问题

* **归一化**：波段单位是 nm；Stokes 矢量在 `wing` 波长处归一，喂给反演的应是
  连续谱归一化数据（`I/Ic … V/Ic`）。
* **波长参考**：`MEForward` 计算 `u = (λ − lambda0)/Dlambda_D`，所以 `lambda0`
  必须是你希望 `v_los` 相对于的线心。内置 `sdo_hmi` 模型会对此做校验（第 6 节）。
* **数据类型**：工作精度是 float32（退火器内部会转换）；float64 也支持，且梯形
  参考实现会更准。
* **`PI2NN` 与 `nn.Module`**：`PI2NN.train` 是训练循环，因此**不要**对 `PI2NN`
  实例用 `.eval()`（见 5.4 节）。加载内置权重不会触发这一点，但你自己的代码可能。
* **`method='csa'`** 曾因 `UnboundLocalError: E_init` 必崩，现已修复
  （`CudaAnnealing._annealing`）；仍推荐 `'gsa'`。
* **`initial_guess.py` 的加载**：内置 checkpoint 是脚本以*顶层*模块 `PI2NN`
  保存的，直接 `torch.load` 无法反序列化，`load_pi2nn_model` 会把那些类引用映射
  到 `CuSP.PI2NN`。`torch>=2.6` 还需要 `weights_only=False`，加载器在支持时会自动传。

## 10. 复现验证

上文引用的数字来自与本仓库配套的脚本（作者工作区的 `../analysis/`）：

| 脚本 | 检查内容 |
|---|---|
| `verify_voigt_port.py` | Voigt 线型对照 `scipy.special.wofz` 与改动前的代码（用 `git show HEAD:` 取出），以及形状/数据类型/自动微分与加速比 |
| `probe_rational_domain.py` | 有理逼近的有效域（`u` 的符号、`a → 0`、大 `|u|`、float32） |
| `inspect_pi2nn_pkl.py` | 静态检查 checkpoint 里的类引用 |
| `probe_pi2nn_model.py` | 读取 checkpoint 中保存的 `bounds` / `norm_scale` / `forward_model` |
| `validate_pi2nn_initial_guess.py` | `initial_guess='sdo_hmi'` 的端到端验证：网络精度、退火改善、配置守卫 |
| `smoke_pi2nn.py` | `PI2NN` 训练冒烟测试（`dense_spectrum` 两条路径） |
| `demo_initial_guess_sdo_hmi.py` | 上文 `initial_guess='sdo_hmi'` 的用法演示 |
| `verify_readme_snippets.py` | 把本 README 里的每段代码原样跑一遍，保证文档里的 API 不会与实现脱节 |

## 11. 引用

如果本代码对你的研究有帮助，请引用配套论文：

> G. Y. Chen, Y. Guo, C. J. Díaz Baso, Q. Hao & M. D. Ding,
> *A synergistic spectropolarimetric inversion via gradient-bias annealing and
> physics-informed neural networks*,
> Astronomy & Astrophysics **710**, A227 (2026).
> DOI: [10.1051/0004-6361/202659270](https://doi.org/10.1051/0004-6361/202659270)
> （开放获取，CC BY 4.0）

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

**作者：** Chen Guoyin · gychen@smail.nju.edu.cn · 版本 0.1.0

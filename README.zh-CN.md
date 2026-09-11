# CuSP — Cuda-supported SpectroPolarimetry

[English](README.md) | **中文**

> **技术文章。** `CuSP` 对应论文 *A synergistic spectropolarimetric inversion via
> gradient-bias annealing and physics-informed neural networks*，作者
> G. Y. Chen、Y. Guo、C. J. Díaz Baso、Q. Hao、M. D. Ding，
> **A&A 710, A227 (2026)**，DOI
> [10.1051/0004-6361/202659270](https://doi.org/10.1051/0004-6361/202659270)
> （开放获取，CC BY 4.0）。使用本代码时请引用该文；可直接复制的 BibTeX 见
> [第 10 节](#10-引用)。

`CuSP` 是一个基于 PyTorch 的**太阳偏振光谱 Milne–Eddington（ME）反演**工具包。
它提供完全可微的 Stokes `IQUV` 前向模型、GPU/CPU 上的批量反演（CMA-ES、广义模拟
退火、Levenberg–Marquardt）、快速的 Voigt 线型计算，以及一个物理信息神经网络
（`PI2NN`）——可以据此给反演一个良好的初猜，而不是从随机噪声出发。

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
10. [引用](#10-引用)

---

## 1. 功能概览

| 功能 | 模块 | 说明 |
|---|---|---|
| ME 前向合成 Stokes `IQUV` | `me_forward.py` | Unno–Rachkovsky 解析解，全向量化、可微、与设备无关 |
| ME 反演（四种方法可互换） | `me_inversion.py` | `method='cmaes'`（CMA-ES，**默认**）、`'annealing'`（带梯度偏置步长选择的广义模拟退火，即论文的 **GBA**）、`'lm'`（Levenberg–Marquardt）或 `'csa'`（旧的共轭模拟退火）；支持 GPU |
| 快速 Voigt / Faraday–Voigt 线型 | `voigt.py` | 用 Faddeeva 函数的 7/7 复有理逼近——不做数值积分，小阻尼到大阻尼都准确 |
| 向量化响应函数（Jacobian） | `me_rf.py` | 按（像素 × 波长）逐元素自动微分：四个 Stokes 分量各一次反向传播就能得到响应函数，`lm` 的 Jacobian 不再随波长采样数增长 |
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
    ├── lm.py                # BatchLM        ：批量 Levenberg-Marquardt
    ├── cmaes.py             # BatchCMAES     ：批量 CMA-ES
    ├── me_inverters.py      # 把 BatchLM / BatchCMAES 接到 ME 问题上
    ├── me_rf.py             # 向量化响应函数（lm 的 Jacobian）
    ├── voigt.py             # 快速 Voigt 与 Faraday–Voigt 线型
    ├── initial_guess.py     # PI2NN 初猜（加载 / 注册 / 推理）
    ├── PI2NN.py             # 物理信息神经网络反演
    ├── data/
    │   └── pi2nn_sdo_hmi.pkl  # 面向 SDO/HMI 训练好的 PI2NN 权重
    └── examples/
        ├── forward.ipynb    # 前向模型教程
        ├── inversion.ipynb  # 反演教程
        ├── inversion_random_cmaes_lm.py   # 随机初值 -> CMA-ES -> LM
        └── inversion_pi2nn_lm.py          # PI2NN 初值 -> LM
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

四种优化器作用在**完全相同**的归一化参数盒 `[0,1]^8` 上，接受相同的
`initial_guess`、最小化同一个加权 chi2，因此换一个关键字就能切换或对比：

| `method=` | 优化器 | 特点 |
|---|---|---|
| `'cmaes'` | CMA-ES（`CuSP.cmaes.BatchCMAES`） | 无梯度的全局搜索，不需要 Jacobian——**默认方法** |
| `'annealing'`（别名 `'gsa'`） | 带梯度偏置步长选择的广义模拟退火（`DualAnnealing`），即论文的 **GBA** | 随机、无需梯度、对差初值较稳健 |
| `'lm'` | Levenberg–Marquardt（`CuSP.lm.BatchLM`） | 局部方法，Jacobian 由向量化响应函数（`me_rf.py`，每个 Stokes 分量一次反向传播）精确自动微分得到；**好初值下**收敛最快最准 |
| `'csa'` | 共轭模拟退火（`CudaAnnealing`，旧） | 会打印完整 epoch 日志 |

```python
from CuSP import MEInversion

inv = MEInversion(torch.tensor(lm).float(), landeG=2.5, lambda0=630.25, wing=wing)
x0 = inv.make_initial_guess(iquv_obs, initial_guess='random')   # 或 'sdo_hmi'，见第 6 节

# (a) CMA-ES —— 默认方法，不用给关键字
params = inv(iquv_obs, max_iter=200)

# (b) 随机初猜 + 广义/梯度偏置模拟退火（即论文的 GBA）
params = inv(iquv_obs, method='annealing', maxiter=200, initial_temp=5230.)

# (c) Levenberg-Marquardt（局部方法，要给好初值）
params = inv(iquv_obs, method='lm', max_iter=60, initial_guess=x0)

# (d) 旧的共轭模拟退火
params = inv(iquv_obs, method='csa', max_iter=1000)

# (e) 用 PI2NN 预测作初值 —— 内置网络绑定 SDO/HMI 采样，需要单独的反演对象：见第 6 节
#     params = inv_hmi(iquv_obs_hmi, initial_guess='sdo_hmi', method='lm')
```

该用哪个？

* `cmaes`（默认）是无梯度的全局搜索：不需要 Jacobian，单位精度的代价低，适合作为大批量
  数据的第一遍；
* `annealing` 是论文的 GBA 驱动——随机、无需梯度、对差初值较稳健，但需要把
  `initial_temp` 调到与你的 merit 量级匹配；
* `lm` 是局部方法，一旦有好初值就是四者里最准的，而向量化响应函数（见下）让它的每次
  迭代都很便宜；
* `csa` 仅为向后兼容保留。

⚠️ `initial_temp` 必须与你问题的 merit 量级（`merit_function` 返回的 `chi2/dof`）匹配。
如果它比拟合能达到的数值高很多，几乎所有试探都会被接受、退火基本不动，因此请把它设成
与你 `chi2/dof` 同一数量级。

#### 推荐的使用流程

* **有网络提供初猜**（例如 `initial_guess='sdo_hmi'`）→ 直接用 **`lm`**。
  它是局部方法，只要初值落在正确的盆地就能精确收敛，是四者里最准的。
* **没有网络（随机初猜）** → **先跑 `cmaes` 或 `annealing` 做全局搜索，再交给
  `lm` 精修**：在这个多峰问题上，`lm` 单独从随机初值出发找不到盆地；`cmaes`
  就是默认方法，所以直接 `inv(iquv_obs)` 已经跑完那个全局段。

```python
import torch

# —— 没有网络：先全局搜索，再用 LM 精修 ——
p1 = inv(iquv_obs, method='cmaes', max_iter=200)
# p1 = inv(iquv_obs, method='annealing', maxiter=100, initial_temp=0.1)
x1 = inv.normalizing_parameter(torch.as_tensor(p1))     # 换回归一化 [0,1]^8
p2 = inv(iquv_obs, method='lm', max_iter=60, initial_guess=x1)

# —— 有网络给初猜：直接用 LM ——
#（内置网络绑定它自己的波长配置，需要用对应的反演对象，见第 6 节）
p2 = inv_hmi(iquv_obs_hmi, method='lm', max_iter=60, initial_guess='sdo_hmi')
```

两条路线的完整可运行版本见 `src/CuSP/examples/inversion_random_cmaes_lm.py` 与
`src/CuSP/examples/inversion_pi2nn_lm.py`。

因此没有网络时推荐「`cmaes`/`annealing` 全局段 + `lm` 精修」这条路线：全局段负责把
局部优化带进正确的盆地。两点注意：

* 全局段必须给够探索能力，因此 `CuSP.me_inverters.run_cmaes` 的默认是
  `pop_size=30, bounded=False`（更大种群、不做 logit 压缩），而不是类的默认值；
* `cmaes` 单位精度的代价最低，因此上百万像素时适合做第一遍，再对需要更高精度的地方补 `lm`。

#### 响应函数（Jacobian）

`method='lm'` 需要 Stokes 矢量对八个参数的导数。ME 前向模型在像素和波长两个维度上都是
逐元素独立的，因此 `CuSP.me_rf` 把参数沿一条新的波长轴展开、把展开后的张量标记为 autograd
叶子节点，然后**每个 Stokes 分量只做一次反向传播**——对 `4N` 个观测量逐个反向的循环消失了，
代价不再随波长采样数增长。

```python
from CuSP import VectorizedResponseFunction

rf = VectorizedResponseFunction(inv)   # 传 MEForward 或 MEInversion 实例
rf(x0)                                 # (B, N, 4, 8)  = d(Stokes)/d(归一化参数)
rf(x0, layout='lm')                    # (B, 4*N, 8)  = BatchLM 需要的 Jacobian
```

`run_lm` 通过 BatchLM 的 `usrLMcoef` 钩子把它装成默认实现（`lm.py` 本身不用改）；
传 `rf_method='loop'` 可以退回原来"逐个观测量反向"的 Jacobian。

收益是结构性的而不是渐进优化：逐个观测量方法每次 LM 系数计算要 `4N + 1` 次反向传播
（每个观测量一次 + 损失梯度一次），而向量化方法固定 **5** 次——4 个 Stokes 分量各一次，
再加连续谱归一化一次——与波长采样数 `N` 无关。因此密集采样不会比 HMI 六点更贵。
（单像素在 GPU 上仍然是启动开销主导，所以 GPU 要用在批量上。）

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
| `method` | `'cmaes'` | `'cmaes'`、`'annealing'`（= `'gsa'`）、`'lm'` 或 `'csa'` |
| `initial_guess` | `None` | 见第 6 节 |
| `x_guess` | `None` | 显式的归一化 `(B,8)` 起点，优先于 `initial_guess` |
| `maxiter` / `max_iter` | `200` | 迭代数：`cmaes` 200、`lm` 60；退火方法则是每个温度下的步数（1000） |
| `initial_temp` | `5230.` | 起始温度 |
| `visit`、`accept`、`no_local_search` | `2.62`、`-5.0`、`False` | `gsa` 的控制参数 |
| `adam` | `{}`，即 `nepoch=1000` | 退火之后跑的局部 Adam 精修；传 `adam=dict(nepoch=0)` 可跳过，`dict(nepoch=2000, learning_rate=1e-3)` 可调节 |
| `max_batches` | `1e10` | 大批量按此尺寸分块 |
| `rf_method` | `'vector'` | `method='lm'` 计算 Jacobian 的方式：`'vector'`（`me_rf`，每个 Stokes 分量一次反向）或 `'loop'`（每个观测量一次反向） |
| `objective_clamp` | `(-1., 2.)` | 反归一化前计算目标函数所用的盒（见第 9 节） |
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
* 网络的输出是反演的**起点**而不是最终答案——请按 5.2 节用 `lm`（或退火/CMA-ES 一段）
  继续精修。
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

`Re w` 关于 `u` 为偶、`Im w` 为奇，有理逼近覆盖 `u` 的两个符号以及所有 `a ≥ 0`。

旧梯形实现的 `ynodes` / `lim` 参数仍被接受，但会被忽略。有理逼近按工作数据类型求值，
因此 `|u|` 足够大时 float32 多项式最终会溢出；CuSP 的工作范围远在这个极限之内。

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

**`lm.py`** —— `BatchLM(max_iters, init_damping, damping_mode, step_solver, …)`：
可微 `forward(x)` 上的批量 Levenberg–Marquardt，`__call__(Y, guess, sig=…) -> x_best`。
**`cmaes.py`** —— `BatchCMAES(max_iters, pop_size, init_sigma, patience, …)`：
批量 CMA-ES（对角/完整协方差），调用约定相同。
**`me_inverters.py`** —— `MEObjective`（展平的加权 chi2 目标函数）、
`run_lm(..., rf_method='vector'|'loop')`、`run_cmaes`：把两个优化器接到 ME 问题上的
适配层（同一个归一化盒、与 `merit_function` 相同的 `sig`）。

**`me_rf.py`** —— `VectorizedResponseFunction(forward, wavelengths=None, wing=None,
zoom_factor=1.0)`：逐元素响应函数，
`__call__(x, normalized=True, layout='nw48'|'lm', return_stokes=False)` 返回
`(B, N, 4, 8)` 或展平的 `(B, 4N, 8)` Jacobian；`vectorized_response_function` 是
一次性调用封装，`make_vectorized_lmcoef(objective, ...)` 是 `run_lm` 默认使用的
`_LMcoef` 兼容钩子。

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
  （`CudaAnnealing._annealing`）；`'gsa'`/`'annealing'` 仍是论文的 GBA 驱动，
  但默认方法是 `'cmaes'`。
* **merit 数值**：`ivs_results['e']` / `['e0']` 一律由返回的参数（`x` / `x0`）
  重新计算，因为退火器内部的 `e_best` 可能和它返回的 `x_best` 不同步。
* **LM 的 Jacobian**：`run_lm` 默认用 `me_rf.VectorizedResponseFunction` 求 Jacobian
  （每个 Stokes 分量一次反向传播，代价与波长采样数无关），而不是逐个观测量反向；
  `rf_method='loop'` 可退回旧行为。两者都是 `MEObjective.forward` 的精确导数，即
  *包含* clamp 余量的目标函数的导数：落在 `objective_clamp` 之外的参数其 Jacobian
  列为 0，与旧实现完全一致。另外 BatchLM 里 pyPRT 专用的软边界惩罚项，在 `run_lm`
  配置的单组 `decomposition=[8]` 下恒为 0。
* **`initial_guess.py` 的加载**：内置 checkpoint 是脚本以*顶层*模块 `PI2NN`
  保存的，直接 `torch.load` 无法反序列化，`load_pi2nn_model` 会把那些类引用映射
  到 `CuSP.PI2NN`。`torch>=2.6` 还需要 `weights_only=False`，加载器在支持时会自动传。

## 10. 引用

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

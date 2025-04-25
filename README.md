# CuSP: CUDA-Accelerated Spectropolarimetry Toolkit

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CuSP is a high-performance spectropolarimetry toolkit leveraging CUDA acceleration for efficient forward modeling and inversion calculations.

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA-enabled recommended) or CPU
- PyTorch >=1.13.1

### From PyPI (Recommended)
```bash
pip install CuSP

### From Source
```bash
git clone https://github.com/gychen-NJU/CuSP.git
cd CuSP
pip install -e .

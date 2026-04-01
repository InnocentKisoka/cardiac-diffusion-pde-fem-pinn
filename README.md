# cardiac-diffusion-pde-fem-Physics-Informed Neural Network (PINN)
❤️ FEM + PINN solver for 2D anisotropic cardiac propagation PDE. Classical IMEX-FEM (MATLAB) benchmark vs mesh-free PINN (PyTorch) across 3 diseased diffusion regimes (Σd = 0.1Σh, Σh, 10Σh). Full weak-form derivation, matrix assembly &amp; animations.



# ❤️ Cardiac Wave Propagation: FEM vs PINN on Anisotropic Diffusion-Reaction PDE

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue?logo=mathworks&logoColor=white)](https://mathworks.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**End-to-end hybrid solver** for the nonlinear PDE modeling electrical wave propagation in cardiac tissue with three diseased subdomains (different diffusivity Σd).

> Classical **IMEX Finite Element Method** (MATLAB, Q1 elements, sparse assembly) serves as ground-truth reference.  
> Modern **Physics-Informed Neural Network** (PyTorch) solves the same problem **mesh-free** and generalizes across diffusion regimes with a single architecture.

---

## ✨ Key Highlights
- **PDE**: ∂u/∂t − ∇·(Σ∇u) + f(u) = 0 with cubic reaction f(u) = a(u−fr)(u−ft)(u−fd), Neumann BC, localized initial condition.
- **Three diffusion regimes**:
  - Low (Σd = 0.1Σh) → wave stays localized
  - Normal (Σd = Σh) → smooth propagation
  - High (Σd = 10Σh) → rapid diffusion
- **FEM**: Full weak-form derivation → mass + stiffness matrix assembly → IMEX time stepping → activation time metrics + GIF animations.
- **PINN**: 5-layer tanh network (input: x,y,t) trained on 50k collocation points. Loss = PDE residual + IC + BC.
- **Benchmark**: PINN matches FEM behavior qualitatively across all Σd configs.
- **Team**: Innocent Kisoka & Yassine Oueslati (@usi.ch)

---

## 📐 The Problem (Simplified Monodomain Model)

Domain Ω = (0,1)², diseased regions Ωd1, Ωd2, Ωd3 (circles).  
Σ(x) = Σd in Ωd, Σh elsewhere (Σh = 9.5298×10⁻⁴).  
Reaction: f(u) = 18.515 u (u−0.2383)(u−1).  
Initial pulse in top-right corner.

---

## 🧪 1. Classical FEM Solver (MATLAB)

- **Time scheme**: IMEX (explicit reaction f(ut), implicit diffusion).
- **Space**: Uniform Q1 bilinear elements on M×M grid.
- **Assembly**:
  - Local mass matrix (4×4, 1/36 scaling)
  - Local stiffness matrix with element-wise Σ
  - Global sparse (M+dt·K) system solved at each step.
- **Outputs**:
  - Activation time (first time max(u) > ft)
  - M-matrix check & range validation (u ∈ [0,1])
  - High-quality GIF animations per config
- **Tested**: dt ∈ {0.1, 0.05, 0.025}, ne ∈ {64, 128}, all three Σd.

**Folder**: `FEM_MATLAB/` (run.m + assemble*.m helpers)

---

## 🧠 2. Physics-Informed Neural Network (Python/PyTorch)

**Architecture**
- Input: 3 neurons (x, y, t)
- Hidden: 5 layers [200, 100, 100, 50, 50] + tanh
- Output: 1 neuron u(x,y,t)

**Loss**
L = L_PDE + L_IC + L_BC  
(50 000 collocation points + extra sampling in diseased regions)

**Training**
- Optimizer: Adam (lr = 1e-3)
- Epochs: 50 000
- Same network for all three Σd configs (zero architecture change!)

**Folder**: `PINN/` (`main.py` + `pinn_solver/` package)

---

## 📊 Results & Visualizations

**FEM reference** (MATLAB):
- Clear difference in propagation speed across Σd values.
- Activation time and solution snapshots exported.

**PINN predictions**:
- Config 1 (low Σd): localized wave
- Config 2 (normal): balanced spread
- Config 3 (high Σd): fast diffusion

All animations and loss curves are saved in `outputs/`.

**Comparison**: PINN qualitatively reproduces FEM behavior without any mesh, proving its power for parametric PDE studies.

---

## 🛠️ How to Run

### FEM (MATLAB)
```matlab
cd FEM_MATLAB
run.m          % loops over all dt / mesh / Σd combos

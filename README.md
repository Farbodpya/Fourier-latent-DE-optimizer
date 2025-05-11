# Fourier-Latent Differential Evolution Optimizer

This repository contains the implementation of a **Fourier-based dimensionality reduction framework** for high-dimensional optimization using **Differential Evolution (DE)** in latent space.  
It accompanies the preprint:

**"Latent-Space Genetic Algorithms for High-Dimensional Optimization via Random Projection, SVD, and Fourier Transform"**  
Zenodo DOI: (https://doi.org/10.5281/zenodo.15380971)

---

## 🔍 Overview

Traditional metaheuristics like Genetic Algorithm (GA) and Differential Evolution (DE) suffer in high-dimensional spaces due to the curse of dimensionality.  
In this work, we project the original 5000D space into a **1D latent space** using **Fourier Transform (FFT/IFT)**, and run DE in this compressed domain.

Despite extreme compression, the optimizer consistently reaches **global optima** for benchmark functions like Sphere, Rastrigin, Ackley, Zakharov, etc.

---

## 🚀 Key Features

- ✅ Unsupervised, training-free dimensionality reduction using **FFT**
- ✅ Fast and accurate optimization in 1D latent space
- ✅ Outperforms classical DE in runtime and convergence
- ✅ Works on 5000-dimensional problems with near-zero final costs
- ✅ Easy to extend with other benchmarks or decompositions (e.g., SVD, RP)

---

## 📈 Benchmark Functions

The following functions are included:

- Sphere  
- Rastrigin  
- Rastrigin II  
- Ackley  
- Griewank  
- Zakharov  

Each is optimized in latent space using `scipy.optimize.differential_evolution`.

---

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Farbodpya/Fourier-latent-DE-optimizer.git
cd Fourier-latent-DE-optimizer

# Run the optimizer
python main.py

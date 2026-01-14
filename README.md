# WCM Surrogate

Surrogate models for efficient parameter estimation in Whole-Cell Models.

Based on the research proposal "Surrogate Models for Efficient Parameter Estimation in Simulation of Whole-Cell Models" (Subr & Weisse, University of Edinburgh).

## Overview

This project implements surrogate models to approximate computationally expensive Whole-Cell Models (WCMs), enabling efficient parameter inference from population-level observations.

## Project Structure

```
wcm-surrogate/
├── src/wcm_surrogate/
│   ├── growth_rate/       # Growth rate model (Thomas et al. 2018)
│   │   └── langevin.py    # Stochastic Langevin equation simulator
│   └── surrogate/         # ML surrogate models (TODO)
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for exploration
└── data/                  # Training data and results
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from wcm_surrogate.growth_rate import GrowthRateSimulator
from wcm_surrogate.growth_rate.langevin import GrowthParameters

# Create simulator with custom parameters
params = GrowthParameters(lambda_0=0.02, k_r=1.5)
sim = GrowthRateSimulator(params, seed=42)

# Run single cell simulation
trajectory = sim.run(t_max=100.0, dt=0.1)

# Run population simulation
population = sim.run_population(n_cells=100, t_max=100.0)
```

## Research Objectives

1. **Growth rate surrogate**: Approximate the Langevin growth model
2. **E. coli WCM surrogate**: Scale to full whole-cell model
3. **Parameter inference**: Estimate parameters from population data
4. **Biological validation**: Apply to antibiotic resistance studies

## References

- Thomas et al. (2018) "Sources, propagation and consequences of stochasticity in cellular growth" Nature Communications
- Skalnik et al. (2023) "Whole-cell modeling of E. coli colonies" PLOS Computational Biology
- Covert Lab WCM: https://github.com/CovertLab/WholeCellEcoliRelease

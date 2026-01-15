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
│   ├── validation/        # Validation against experimental data
│   │   ├── compare.py     # Comparison functions and validation suite
│   │   └── experimental_data.py  # Data loaders for E. coli datasets
│   └── surrogate/         # ML surrogate models (TODO)
├── data/                  # Experimental datasets
│   ├── taheri-araghi_2015_data.csv  # Cell size homeostasis data
│   ├── si_2017_data.csv             # Nutrient-dependent growth data
│   └── scott_2010_data.csv          # Bacterial growth laws data
├── tests/                 # Unit tests
└── notebooks/             # Jupyter notebooks for exploration
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from wcm_surrogate.growth_rate import GrowthRateSimulator
from wcm_surrogate.growth_rate.langevin import GrowthParameters

# Create simulator with custom parameters
# growth_rate: 0.025/min = 1.5/hr, corresponds to ~28 min doubling time
params = GrowthParameters(growth_rate=0.025)
sim = GrowthRateSimulator(params, seed=42)

# Run single cell simulation (2 hours)
trajectory = sim.run(t_max=120.0, dt=0.1)
print(f"Divisions: {len(trajectory['division_times'])}")

# Run population simulation
population = sim.run_population(n_cells=100, t_max=120.0)
print(f"Mean growth rate: {population['mean_growth_rate'].mean() * 60:.2f}/hr")
```

## Running Validation

The validation suite compares simulator output against published E. coli experimental data.

```python
from wcm_surrogate.validation import run_validation_suite

# Run full validation suite
results = run_validation_suite(verbose=True)

# Check summary
print(f"Passed: {results['summary']['n_passed']}/{results['summary']['n_total']}")
```

Or from command line:

```bash
python -c "from wcm_surrogate.validation import run_validation_suite; run_validation_suite()"
```

### Validation Tests

| Test | Description | Status |
|------|-------------|--------|
| Growth rate calibration | Simulator matches target growth rates | 7/7 pass |
| Growth rate variability | CV of growth rate is biologically realistic | Known limitation |
| Doubling time | Division timing matches expected | Known limitation |

## Experimental Data

The `data/` folder contains curated E. coli datasets from published studies:

| Dataset | Source | Contents |
|---------|--------|----------|
| `taheri-araghi_2015_data.csv` | Taheri-Araghi et al. (2015) | Growth rate, cell volume across nutrient conditions |
| `si_2017_data.csv` | Si et al. (2017) | Cell dimensions, RNA/protein ratios |
| `scott_2010_data.csv` | Scott et al. (2010) | Bacterial growth laws, ribosomal fractions |

Data sourced from [Imperial College coarse-grained model repo](https://github.com/ImperialCollegeLondon/coli-whole-cell-coarse-grained-model).

## Running Tests

```bash
pytest tests/ -v
```

## Research Objectives

1. **Growth rate surrogate**: Approximate the Langevin growth model
2. **E. coli WCM surrogate**: Scale to full whole-cell model
3. **Parameter inference**: Estimate parameters from population data
4. **Biological validation**: Apply to antibiotic resistance studies

## References

- Thomas et al. (2018) "Sources, propagation and consequences of stochasticity in cellular growth" Nature Communications
- Taheri-Araghi et al. (2015) "Cell-Size Control and Homeostasis in Bacteria" Current Biology
- Scott et al. (2010) "Interdependence of Cell Growth and Gene Expression" Science
- Skalnik et al. (2023) "Whole-cell modeling of E. coli colonies" PLOS Computational Biology
- Covert Lab WCM: https://github.com/CovertLab/WholeCellEcoliRelease

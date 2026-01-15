"""
Growth rate model based on Thomas et al. (2018).
"Sources, propagation and consequences of stochasticity in cellular growth"
Nature Communications 9:4528

This implements a stochastic model of cell growth using coupled Langevin equations
that capture:
- Ribosome autocatalysis (ribosomes make more ribosomes)
- Protein synthesis
- Cell volume growth
- Stochastic cell division

Units:
- Time: minutes
- Volume: μm³ (cubic micrometers)
- Counts: number of molecules
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GrowthParameters:
    """Parameters for the growth rate model.

    Default values calibrated to E. coli in rich media (fast growth).

    Biological references:
    - Doubling time in rich media: ~20-25 min
    - Doubling time in minimal media: ~45-60 min
    - Ribosomes per cell: 10,000-70,000 (growth-rate dependent)
    - Proteins per cell: ~2-4 million
    - Cell volume at birth: 0.5-2 μm³
    """
    # Growth rate (per minute)
    # 0.03 /min = 1.8 /hr → ~23 min doubling time (fast growth, rich media)
    # 0.015 /min = 0.9 /hr → ~46 min doubling time (slow growth, minimal media)
    growth_rate: float = 0.025  # /min, corresponds to ~28 min doubling time

    # Ribosome parameters
    # Ribosomes are ~20% of cell mass in fast-growing E. coli
    # Translation rate: ~15-20 aa/s, ribosome makes ~1 protein/min
    ribosome_fraction: float = 0.2      # Fraction of protein that is ribosomal
    ribosome_efficiency: float = 1.0    # Relative translation efficiency

    # Degradation rates (per minute)
    # Ribosomes are very stable (half-life >> cell cycle)
    # Most E. coli proteins are also stable
    gamma_r: float = 0.001   # Ribosome degradation (~700 min half-life)
    gamma_p: float = 0.002   # Protein degradation (~350 min half-life)

    # Division parameters
    # E. coli follows "adder" model: adds constant volume each generation
    division_ratio: float = 2.0   # Divide when V reaches this * V_birth
    CV_division: float = 0.1      # CV of division size (~10%)

    # Noise parameters (dimensionless)
    # These control the coefficient of variation
    noise_ribosome: float = 0.02   # Ribosome production noise
    noise_protein: float = 0.02    # Protein production noise
    noise_growth: float = 0.05     # Growth rate noise (metabolic fluctuations)

    # Initial conditions (at cell birth)
    # Fast-growing E. coli: ~20,000 ribosomes, ~3M proteins, ~1 μm³
    R_0: float = 20000.0    # Ribosome count at birth
    P_0: float = 3.0e6      # Protein count at birth
    V_0: float = 1.0        # Volume at birth (μm³)


@dataclass
class CellState:
    """State of a single cell."""
    R: float        # Ribosome count
    P: float        # Protein count (non-ribosomal)
    V: float        # Cell volume (μm³)
    age: float      # Time since last division (minutes)
    generation: int # Division count
    V_birth: float  # Volume at birth (for adder model)


class GrowthRateSimulator:
    """Simulator for stochastic cell growth dynamics.

    Uses the Euler-Maruyama method to integrate coupled Langevin equations
    describing ribosome autocatalysis, protein synthesis, and cell growth.

    The model captures the key features of bacterial growth:
    1. Exponential growth of cell volume
    2. Growth rate depends on ribosome concentration
    3. Stochastic gene expression adds cell-to-cell variability
    4. Division occurs when cells reach approximately 2x birth size

    Example:
        >>> params = GrowthParameters(growth_rate=0.025)  # ~28 min doubling
        >>> sim = GrowthRateSimulator(params)
        >>> trajectory = sim.run(t_max=120.0)  # 2 hours
        >>> print(f"Divisions: {len(trajectory['division_times'])}")
    """

    def __init__(self, params: Optional[GrowthParameters] = None, seed: Optional[int] = None):
        """Initialize the simulator.

        Args:
            params: Growth model parameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.params = params or GrowthParameters()
        self.rng = np.random.default_rng(seed)

    def _instantaneous_growth_rate(self, state: CellState) -> float:
        """Calculate instantaneous growth rate with noise.

        Growth rate fluctuates around the target value due to
        metabolic and environmental noise.

        This simpler model directly controls growth rate rather than
        deriving it from ribosome concentration, making the dynamics
        more stable and predictable.
        """
        p = self.params

        # Base growth rate with multiplicative noise
        # Using Ornstein-Uhlenbeck-like bounded fluctuations
        noise = p.noise_growth * self.rng.standard_normal()

        # Growth rate fluctuates around target
        rate = p.growth_rate * (1 + noise)

        # Ensure non-negative and bounded
        return np.clip(rate, 0.001, p.growth_rate * 3)

    def _drift(self, state: CellState, growth_rate: float) -> tuple[float, float, float]:
        """Calculate drift terms for the Langevin equations.

        The model follows bacterial growth laws:
        - Ribosomes are autocatalytic (make more ribosomes)
        - Protein production scales with ribosome count
        - Both are diluted by cell growth

        At steady state (balanced growth):
        - dR/dt = 0: production = dilution + degradation
        - dP/dt = 0: production = dilution + degradation
        - dV/dt = λV: exponential volume growth

        Returns:
            (dR/dt, dP/dt, dV/dt) drift terms
        """
        p = self.params

        # Ribosome production rate
        # At steady state: k_r * R = (λ + γ_r) * R, so k_r = λ + γ_r
        k_r = growth_rate + p.gamma_r

        # Protein production rate
        # At steady state: k_p * R = (λ + γ_p) * P
        # With P/R = P_0/R_0, we get k_p = (λ + γ_p) * (P_0/R_0)
        k_p = (growth_rate + p.gamma_p) * (p.P_0 / p.R_0)

        # Ribosome dynamics: production - degradation - dilution
        # Production is k_r * R (ribosomes make ribosomes)
        dR = k_r * state.R - p.gamma_r * state.R - growth_rate * state.R

        # Protein dynamics: production - degradation - dilution
        # Production is k_p * R (ribosomes make proteins)
        dP = k_p * state.R - p.gamma_p * state.P - growth_rate * state.P

        # Volume growth (exponential)
        dV = growth_rate * state.V

        return dR, dP, dV

    def _diffusion(self, state: CellState, growth_rate: float) -> tuple[float, float]:
        """Calculate diffusion (noise) terms.

        Noise in gene expression follows Poisson statistics:
        variance ~ mean, so std ~ sqrt(mean)

        Returns:
            (noise_R, noise_P) diffusion coefficients
        """
        p = self.params

        # Production rates (same as in drift)
        k_r = growth_rate + p.gamma_r
        k_p = (growth_rate + p.gamma_p) * (p.P_0 / p.R_0)

        # Noise scales with sqrt of production (Poisson)
        # Scaled by noise parameters to control CV
        # Higher noise_ribosome/noise_protein → more cell-to-cell variability
        noise_R = p.noise_ribosome * np.sqrt(max(k_r * state.R, 1.0))
        noise_P = p.noise_protein * np.sqrt(max(k_p * state.R, 1.0))

        return noise_R, noise_P

    def _check_division(self, state: CellState) -> bool:
        """Check if cell should divide.

        E. coli follows an "adder" model: cells add a roughly constant
        volume each generation, regardless of birth size.

        Division occurs when V ≈ 2 * V_birth (with noise).
        """
        p = self.params

        # Target division volume (with noise)
        noise = 1 + p.CV_division * self.rng.standard_normal()
        V_division = state.V_birth * p.division_ratio * max(noise, 0.7)

        return state.V >= V_division

    def _divide(self, state: CellState) -> CellState:
        """Perform cell division.

        Molecules are partitioned binomially between daughter cells.
        We track one daughter (randomly chosen).

        Division is approximately symmetric with small variations.
        """
        # Partition fraction (slightly asymmetric)
        frac = 0.5 + 0.02 * self.rng.standard_normal()
        frac = np.clip(frac, 0.45, 0.55)

        new_V = state.V * frac

        return CellState(
            R=state.R * frac,
            P=state.P * frac,
            V=new_V,
            age=0.0,
            generation=state.generation + 1,
            V_birth=new_V  # This daughter's birth volume
        )

    def step(self, state: CellState, dt: float) -> tuple[CellState, bool, float]:
        """Advance simulation by one time step using Euler-Maruyama.

        Args:
            state: Current cell state
            dt: Time step size (minutes)

        Returns:
            (new_state, divided, growth_rate): Updated state, division flag, and growth rate
        """
        # Calculate growth rate
        growth_rate = self._instantaneous_growth_rate(state)

        # Calculate drift and diffusion
        dR, dP, dV = self._drift(state, growth_rate)
        noise_R, noise_P = self._diffusion(state, growth_rate)

        # Generate Wiener increments
        dW = self.rng.standard_normal(2) * np.sqrt(dt)

        # Euler-Maruyama update
        new_R = state.R + dR * dt + noise_R * dW[0]
        new_P = state.P + dP * dt + noise_P * dW[1]
        new_V = state.V + dV * dt

        # Ensure non-negative values with reasonable minimums
        new_R = max(new_R, 100.0)     # At least 100 ribosomes
        new_P = max(new_P, 10000.0)   # At least 10k proteins
        new_V = max(new_V, 0.1)       # At least 0.1 μm³

        new_state = CellState(
            R=new_R,
            P=new_P,
            V=new_V,
            age=state.age + dt,
            generation=state.generation,
            V_birth=state.V_birth
        )

        # Check for division
        divided = False
        if self._check_division(new_state):
            new_state = self._divide(new_state)
            divided = True

        return new_state, divided, growth_rate

    def run(
        self,
        t_max: float = 120.0,
        dt: float = 0.1,
        record_interval: int = 1
    ) -> dict[str, np.ndarray]:
        """Run the simulation.

        Args:
            t_max: Total simulation time (minutes)
            dt: Integration time step (minutes)
            record_interval: Record state every N steps

        Returns:
            Dictionary with time series of all state variables:
            - time: Time points (minutes)
            - ribosomes: Ribosome counts
            - proteins: Protein counts
            - volume: Cell volume (μm³)
            - growth_rate: Instantaneous growth rate (/min)
            - generation: Division count
            - division_times: Times of division events (minutes)
        """
        p = self.params
        n_steps = int(t_max / dt)
        n_records = n_steps // record_interval + 1

        # Initialize storage
        times = np.zeros(n_records)
        ribosomes = np.zeros(n_records)
        proteins = np.zeros(n_records)
        volumes = np.zeros(n_records)
        growth_rates = np.zeros(n_records)
        generations = np.zeros(n_records, dtype=int)
        division_times = []

        # Initial state
        state = CellState(
            R=p.R_0,
            P=p.P_0,
            V=p.V_0,
            age=0.0,
            generation=0,
            V_birth=p.V_0
        )

        # Record initial state
        record_idx = 0
        times[0] = 0.0
        ribosomes[0] = state.R
        proteins[0] = state.P
        volumes[0] = state.V
        growth_rates[0] = self._instantaneous_growth_rate(state)
        generations[0] = state.generation

        # Run simulation
        for i in range(1, n_steps + 1):
            state, divided, growth_rate = self.step(state, dt)

            if divided:
                division_times.append(i * dt)

            if i % record_interval == 0:
                record_idx += 1
                if record_idx < n_records:
                    times[record_idx] = i * dt
                    ribosomes[record_idx] = state.R
                    proteins[record_idx] = state.P
                    volumes[record_idx] = state.V
                    growth_rates[record_idx] = growth_rate
                    generations[record_idx] = state.generation

        return {
            "time": times[:record_idx + 1],
            "ribosomes": ribosomes[:record_idx + 1],
            "proteins": proteins[:record_idx + 1],
            "volume": volumes[:record_idx + 1],
            "growth_rate": growth_rates[:record_idx + 1],
            "generation": generations[:record_idx + 1],
            "division_times": np.array(division_times)
        }

    def run_population(
        self,
        n_cells: int = 100,
        t_max: float = 120.0,
        dt: float = 0.1
    ) -> dict[str, np.ndarray]:
        """Run simulation for a population of cells.

        Each cell is simulated independently with its own stochastic trajectory.
        Useful for generating training data for surrogate models.

        Args:
            n_cells: Number of independent cells to simulate
            t_max: Simulation time per cell (minutes)
            dt: Time step (minutes)

        Returns:
            Dictionary with population-level statistics
        """
        final_growth_rates = []
        mean_growth_rates = []
        division_times_list = []
        mean_volumes = []

        for i in range(n_cells):
            # Each cell gets different random seed
            cell_sim = GrowthRateSimulator(self.params, seed=self.rng.integers(0, 2**31))
            traj = cell_sim.run(t_max=t_max, dt=dt, record_interval=10)

            final_growth_rates.append(traj["growth_rate"][-1])
            mean_growth_rates.append(np.mean(traj["growth_rate"]))
            mean_volumes.append(np.mean(traj["volume"]))

            # Collect inter-division times
            if len(traj["division_times"]) > 1:
                interdiv = np.diff(traj["division_times"])
                division_times_list.extend(interdiv.tolist())

        return {
            "final_growth_rate": np.array(final_growth_rates),
            "mean_growth_rate": np.array(mean_growth_rates),
            "mean_volume": np.array(mean_volumes),
            "interdivision_times": np.array(division_times_list) if division_times_list else np.array([]),
            "n_cells": n_cells,
        }


def generate_training_data(
    n_samples: int = 1000,
    param_ranges: Optional[dict[str, tuple[float, float]]] = None,
    t_max: float = 120.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate training data for surrogate model.

    Samples parameters uniformly from specified ranges and runs simulations
    to collect input-output pairs.

    Args:
        n_samples: Number of parameter samples
        param_ranges: Dict mapping parameter names to (min, max) tuples.
                     Defaults to biologically plausible ranges.
        t_max: Simulation time per sample (minutes)
        seed: Random seed

    Returns:
        (X, y, param_names):
            X has shape (n_samples, n_params) - input parameters
            y has shape (n_samples, n_outputs) - output statistics
            param_names - list of parameter names
    """
    rng = np.random.default_rng(seed)

    # Default parameter ranges (biologically plausible for E. coli)
    if param_ranges is None:
        param_ranges = {
            "growth_rate": (0.010, 0.040),      # 0.6-2.4 /hr (slow to fast growth)
            "noise_ribosome": (0.005, 0.05),    # Low to moderate noise
            "noise_protein": (0.005, 0.05),
            "noise_growth": (0.02, 0.10),
        }

    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # Sample parameters
    X = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low, high = param_ranges[name]
        X[:, i] = rng.uniform(low, high, n_samples)

    # Run simulations
    outputs = []
    for i in range(n_samples):
        # Create parameter object
        param_dict = {name: X[i, j] for j, name in enumerate(param_names)}
        params = GrowthParameters(**param_dict)

        # Run simulation
        sim = GrowthRateSimulator(params, seed=rng.integers(0, 2**31))
        traj = sim.run(t_max=t_max, dt=0.1, record_interval=10)

        # Extract outputs
        mean_gr = np.mean(traj["growth_rate"])
        std_gr = np.std(traj["growth_rate"])
        cv_gr = std_gr / mean_gr if mean_gr > 0 else 0

        # Division rate (divisions per minute)
        n_divisions = len(traj["division_times"])
        div_rate = n_divisions / t_max

        # Mean doubling time (if we have divisions)
        if n_divisions > 1:
            interdiv = np.diff(traj["division_times"])
            mean_doubling = np.mean(interdiv)
        else:
            mean_doubling = np.log(2) / mean_gr if mean_gr > 0 else np.inf

        outputs.append([
            mean_gr * 60,           # Convert to /hr for comparison with experiments
            cv_gr,                  # CV of growth rate
            mean_doubling,          # Mean doubling time (min)
            np.mean(traj["volume"]) # Mean cell volume
        ])

    y = np.array(outputs)

    return X, y, param_names

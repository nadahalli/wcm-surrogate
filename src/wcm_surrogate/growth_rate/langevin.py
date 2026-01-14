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
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GrowthParameters:
    """Parameters for the growth rate model.

    Based on Table 1 from Thomas et al. (2018).
    Default values are representative of E. coli growth.
    """
    # Transcription/translation rates
    k_r: float = 1.0        # Ribosome production rate (per ribosome per time)
    k_p: float = 1.0        # Protein production rate
    gamma_r: float = 0.01   # Ribosome degradation rate
    gamma_p: float = 0.01   # Protein degradation rate

    # Growth parameters
    lambda_0: float = 0.02  # Base growth rate (per minute)

    # Division parameters
    V_div: float = 2.0      # Division volume (relative to birth volume)
    CV_div: float = 0.1     # Coefficient of variation for division

    # Noise parameters
    sigma_r: float = 0.1    # Ribosome production noise
    sigma_p: float = 0.1    # Protein production noise

    # Initial conditions
    R_0: float = 1000.0     # Initial ribosome count
    P_0: float = 5000.0     # Initial protein count
    V_0: float = 1.0        # Initial cell volume


@dataclass
class CellState:
    """State of a single cell."""
    R: float        # Ribosome count
    P: float        # Protein count
    V: float        # Cell volume
    age: float      # Time since last division
    generation: int # Division count


class GrowthRateSimulator:
    """Simulator for stochastic cell growth dynamics.

    Uses the Euler-Maruyama method to integrate the coupled Langevin equations
    describing ribosome autocatalysis, protein synthesis, and cell growth.

    Example:
        >>> params = GrowthParameters(lambda_0=0.02)
        >>> sim = GrowthRateSimulator(params)
        >>> trajectory = sim.run(t_max=100.0, dt=0.01)
    """

    def __init__(self, params: Optional[GrowthParameters] = None, seed: Optional[int] = None):
        """Initialize the simulator.

        Args:
            params: Growth model parameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.params = params or GrowthParameters()
        self.rng = np.random.default_rng(seed)

    def _growth_rate(self, state: CellState) -> float:
        """Calculate instantaneous growth rate.

        Growth rate depends on ribosome concentration (R/V).
        """
        p = self.params
        ribosome_concentration = state.R / state.V
        return p.lambda_0 * ribosome_concentration / (p.R_0 / p.V_0)

    def _drift(self, state: CellState) -> tuple[float, float, float]:
        """Calculate drift terms for the Langevin equations.

        Returns:
            (dR/dt, dP/dt, dV/dt) drift terms
        """
        p = self.params
        growth_rate = self._growth_rate(state)

        # Ribosome dynamics: production - degradation - dilution
        dR = p.k_r * state.R - p.gamma_r * state.R - growth_rate * state.R

        # Protein dynamics: production - degradation - dilution
        dP = p.k_p * state.R - p.gamma_p * state.P - growth_rate * state.P

        # Volume growth
        dV = growth_rate * state.V

        return dR, dP, dV

    def _diffusion(self, state: CellState) -> tuple[float, float]:
        """Calculate diffusion (noise) terms.

        Returns:
            (noise_R, noise_P) diffusion coefficients
        """
        p = self.params
        # Noise scales with sqrt of production rate (Poisson statistics)
        noise_R = p.sigma_r * np.sqrt(max(p.k_r * state.R, 0))
        noise_P = p.sigma_p * np.sqrt(max(p.k_p * state.R, 0))
        return noise_R, noise_P

    def _check_division(self, state: CellState) -> bool:
        """Check if cell should divide.

        Division occurs stochastically when volume exceeds threshold.
        Uses a size-based adder model with noise.
        """
        p = self.params
        # Division threshold with noise
        V_threshold = p.V_div * p.V_0 * (1 + p.CV_div * self.rng.standard_normal())
        return state.V >= V_threshold

    def _divide(self, state: CellState) -> CellState:
        """Perform cell division.

        Molecules are partitioned binomially between daughter cells.
        We track one daughter (randomly chosen).
        """
        # Binomial partitioning (approximate with normal for large counts)
        frac = 0.5 + 0.05 * self.rng.standard_normal()  # Slight asymmetry
        frac = np.clip(frac, 0.3, 0.7)

        return CellState(
            R=state.R * frac,
            P=state.P * frac,
            V=state.V * frac,
            age=0.0,
            generation=state.generation + 1
        )

    def step(self, state: CellState, dt: float) -> tuple[CellState, bool]:
        """Advance simulation by one time step using Euler-Maruyama.

        Args:
            state: Current cell state
            dt: Time step size

        Returns:
            (new_state, divided): Updated state and whether division occurred
        """
        # Calculate drift and diffusion
        dR, dP, dV = self._drift(state)
        noise_R, noise_P = self._diffusion(state)

        # Generate Wiener increments
        dW = self.rng.standard_normal(2) * np.sqrt(dt)

        # Euler-Maruyama update
        new_R = state.R + dR * dt + noise_R * dW[0]
        new_P = state.P + dP * dt + noise_P * dW[1]
        new_V = state.V + dV * dt

        # Ensure non-negative values
        new_R = max(new_R, 1.0)
        new_P = max(new_P, 1.0)
        new_V = max(new_V, 0.1)

        new_state = CellState(
            R=new_R,
            P=new_P,
            V=new_V,
            age=state.age + dt,
            generation=state.generation
        )

        # Check for division
        divided = False
        if self._check_division(new_state):
            new_state = self._divide(new_state)
            divided = True

        return new_state, divided

    def run(
        self,
        t_max: float = 100.0,
        dt: float = 0.1,
        record_interval: int = 1
    ) -> dict[str, np.ndarray]:
        """Run the simulation.

        Args:
            t_max: Total simulation time
            dt: Integration time step
            record_interval: Record state every N steps

        Returns:
            Dictionary with time series of all state variables
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
            generation=0
        )

        # Record initial state
        record_idx = 0
        times[0] = 0.0
        ribosomes[0] = state.R
        proteins[0] = state.P
        volumes[0] = state.V
        growth_rates[0] = self._growth_rate(state)
        generations[0] = state.generation

        # Run simulation
        for i in range(1, n_steps + 1):
            state, divided = self.step(state, dt)

            if divided:
                division_times.append(i * dt)

            if i % record_interval == 0:
                record_idx += 1
                if record_idx < n_records:
                    times[record_idx] = i * dt
                    ribosomes[record_idx] = state.R
                    proteins[record_idx] = state.P
                    volumes[record_idx] = state.V
                    growth_rates[record_idx] = self._growth_rate(state)
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
        t_max: float = 100.0,
        dt: float = 0.1
    ) -> dict[str, np.ndarray]:
        """Run simulation for a population of cells.

        Useful for generating training data for surrogate models.

        Args:
            n_cells: Number of independent cells to simulate
            t_max: Simulation time per cell
            dt: Time step

        Returns:
            Dictionary with population-level statistics
        """
        final_growth_rates = []
        mean_growth_rates = []
        division_counts = []
        mean_volumes = []

        for _ in range(n_cells):
            traj = self.run(t_max=t_max, dt=dt, record_interval=10)

            final_growth_rates.append(traj["growth_rate"][-1])
            mean_growth_rates.append(np.mean(traj["growth_rate"]))
            division_counts.append(len(traj["division_times"]))
            mean_volumes.append(np.mean(traj["volume"]))

        return {
            "final_growth_rate": np.array(final_growth_rates),
            "mean_growth_rate": np.array(mean_growth_rates),
            "division_count": np.array(division_counts),
            "mean_volume": np.array(mean_volumes)
        }


def generate_training_data(
    n_samples: int = 1000,
    param_ranges: Optional[dict[str, tuple[float, float]]] = None,
    t_max: float = 50.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data for surrogate model.

    Samples parameters uniformly from specified ranges and runs simulations
    to collect input-output pairs.

    Args:
        n_samples: Number of parameter samples
        param_ranges: Dict mapping parameter names to (min, max) tuples
        t_max: Simulation time per sample
        seed: Random seed

    Returns:
        (X, y): Parameter array and corresponding outputs
            X has shape (n_samples, n_params)
            y has shape (n_samples, n_outputs)
    """
    rng = np.random.default_rng(seed)

    # Default parameter ranges (biologically plausible)
    if param_ranges is None:
        param_ranges = {
            "lambda_0": (0.005, 0.05),
            "k_r": (0.5, 2.0),
            "k_p": (0.5, 2.0),
            "sigma_r": (0.01, 0.3),
            "sigma_p": (0.01, 0.3),
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
        sim = GrowthRateSimulator(params, seed=seed + i if seed else None)
        traj = sim.run(t_max=t_max, dt=0.1, record_interval=10)

        # Extract outputs
        outputs.append([
            np.mean(traj["growth_rate"]),
            np.std(traj["growth_rate"]),
            len(traj["division_times"]) / t_max,  # Division rate
            np.mean(traj["volume"])
        ])

    y = np.array(outputs)

    return X, y, param_names

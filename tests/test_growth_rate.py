"""Tests for the growth rate simulator."""

import numpy as np
import pytest
from wcm_surrogate.growth_rate import GrowthRateSimulator
from wcm_surrogate.growth_rate.langevin import GrowthParameters, generate_training_data


class TestGrowthRateSimulator:
    """Tests for GrowthRateSimulator class."""

    def test_initialization_default_params(self):
        """Simulator initializes with default parameters."""
        sim = GrowthRateSimulator()
        assert sim.params is not None
        assert sim.params.lambda_0 > 0

    def test_initialization_custom_params(self):
        """Simulator accepts custom parameters."""
        params = GrowthParameters(lambda_0=0.05, k_r=2.0)
        sim = GrowthRateSimulator(params)
        assert sim.params.lambda_0 == 0.05
        assert sim.params.k_r == 2.0

    def test_reproducibility_with_seed(self):
        """Same seed produces identical results."""
        params = GrowthParameters()
        sim1 = GrowthRateSimulator(params, seed=42)
        sim2 = GrowthRateSimulator(params, seed=42)

        traj1 = sim1.run(t_max=10.0, dt=0.1)
        traj2 = sim2.run(t_max=10.0, dt=0.1)

        np.testing.assert_array_equal(traj1["ribosomes"], traj2["ribosomes"])
        np.testing.assert_array_equal(traj1["volume"], traj2["volume"])

    def test_run_returns_expected_keys(self):
        """Run returns dictionary with all expected keys."""
        sim = GrowthRateSimulator(seed=42)
        traj = sim.run(t_max=10.0)

        expected_keys = ["time", "ribosomes", "proteins", "volume",
                        "growth_rate", "generation", "division_times"]
        for key in expected_keys:
            assert key in traj, f"Missing key: {key}"

    def test_volume_increases_over_time(self):
        """Cell volume should generally increase (before division)."""
        sim = GrowthRateSimulator(seed=42)
        traj = sim.run(t_max=5.0, dt=0.1)

        # Volume at end should be larger than start (short time, likely no division)
        assert traj["volume"][-1] > traj["volume"][0]

    def test_division_resets_volume(self):
        """Division should reduce cell volume."""
        # Run long enough to ensure division
        params = GrowthParameters(V_div=1.5)  # Lower threshold for faster division
        sim = GrowthRateSimulator(params, seed=42)
        traj = sim.run(t_max=200.0, dt=0.1)

        # Should have at least one division
        assert len(traj["division_times"]) > 0

    def test_positive_values(self):
        """All state variables should remain positive."""
        sim = GrowthRateSimulator(seed=42)
        traj = sim.run(t_max=50.0, dt=0.1)

        assert np.all(traj["ribosomes"] > 0)
        assert np.all(traj["proteins"] > 0)
        assert np.all(traj["volume"] > 0)
        assert np.all(traj["growth_rate"] >= 0)


class TestPopulationSimulation:
    """Tests for population-level simulations."""

    def test_run_population(self):
        """Population simulation returns expected statistics."""
        sim = GrowthRateSimulator(seed=42)
        pop = sim.run_population(n_cells=10, t_max=20.0)

        assert len(pop["final_growth_rate"]) == 10
        assert len(pop["mean_growth_rate"]) == 10
        assert len(pop["division_count"]) == 10

    def test_population_variability(self):
        """Different cells should show variability."""
        sim = GrowthRateSimulator(seed=42)
        pop = sim.run_population(n_cells=20, t_max=30.0)

        # Should have some variance in growth rates
        assert np.std(pop["mean_growth_rate"]) > 0


class TestTrainingDataGeneration:
    """Tests for surrogate training data generation."""

    def test_generate_training_data_shape(self):
        """Training data has correct shape."""
        X, y, param_names = generate_training_data(n_samples=10, t_max=10.0, seed=42)

        assert X.shape[0] == 10
        assert y.shape[0] == 10
        assert X.shape[1] == len(param_names)
        assert y.shape[1] == 4  # 4 output features

    def test_generate_training_data_ranges(self):
        """Generated parameters are within specified ranges."""
        param_ranges = {
            "lambda_0": (0.01, 0.03),
            "k_r": (1.0, 1.5),
        }
        X, y, param_names = generate_training_data(
            n_samples=20,
            param_ranges=param_ranges,
            t_max=5.0,
            seed=42
        )

        lambda_idx = param_names.index("lambda_0")
        k_r_idx = param_names.index("k_r")

        assert np.all(X[:, lambda_idx] >= 0.01)
        assert np.all(X[:, lambda_idx] <= 0.03)
        assert np.all(X[:, k_r_idx] >= 1.0)
        assert np.all(X[:, k_r_idx] <= 1.5)

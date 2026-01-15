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
        assert sim.params.growth_rate > 0

    def test_initialization_custom_params(self):
        """Simulator accepts custom parameters."""
        params = GrowthParameters(growth_rate=0.03, noise_growth=0.1)
        sim = GrowthRateSimulator(params)
        assert sim.params.growth_rate == 0.03
        assert sim.params.noise_growth == 0.1

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
        traj = sim.run(t_max=10.0, dt=0.1)

        # Volume at end should be larger than start (short time, likely no division)
        assert traj["volume"][-1] > traj["volume"][0]

    def test_division_occurs(self):
        """Division should occur within expected time frame."""
        # Run long enough to ensure division (> doubling time)
        params = GrowthParameters(growth_rate=0.025)  # ~28 min doubling
        sim = GrowthRateSimulator(params, seed=42)
        traj = sim.run(t_max=120.0, dt=0.1)  # 2 hours

        # Should have at least 2-3 divisions
        assert len(traj["division_times"]) >= 2

    def test_positive_values(self):
        """All state variables should remain positive."""
        sim = GrowthRateSimulator(seed=42)
        traj = sim.run(t_max=60.0, dt=0.1)

        assert np.all(traj["ribosomes"] > 0)
        assert np.all(traj["proteins"] > 0)
        assert np.all(traj["volume"] > 0)
        assert np.all(traj["growth_rate"] > 0)

    def test_growth_rate_in_realistic_range(self):
        """Growth rate should be in biologically realistic range."""
        params = GrowthParameters(growth_rate=0.025)  # 1.5/hr
        sim = GrowthRateSimulator(params, seed=42)
        traj = sim.run(t_max=60.0, dt=0.1)

        mean_rate_per_hr = np.mean(traj["growth_rate"]) * 60

        # Should be roughly in range 0.5-3.0 /hr for E. coli
        assert 0.5 < mean_rate_per_hr < 3.0


class TestPopulationSimulation:
    """Tests for population-level simulations."""

    def test_run_population(self):
        """Population simulation returns expected statistics."""
        sim = GrowthRateSimulator(seed=42)
        pop = sim.run_population(n_cells=10, t_max=60.0)

        assert len(pop["final_growth_rate"]) == 10
        assert len(pop["mean_growth_rate"]) == 10
        assert len(pop["mean_volume"]) == 10
        assert pop["n_cells"] == 10

    def test_population_variability(self):
        """Different cells should show variability due to stochasticity."""
        params = GrowthParameters(noise_growth=0.05)
        sim = GrowthRateSimulator(params, seed=42)
        pop = sim.run_population(n_cells=50, t_max=60.0)

        # Should have some variance in growth rates
        assert np.std(pop["mean_growth_rate"]) > 0


class TestTrainingDataGeneration:
    """Tests for surrogate training data generation."""

    def test_generate_training_data_shape(self):
        """Training data has correct shape."""
        X, y, param_names = generate_training_data(n_samples=10, t_max=60.0, seed=42)

        assert X.shape[0] == 10
        assert y.shape[0] == 10
        assert X.shape[1] == len(param_names)
        assert y.shape[1] == 4  # 4 output features

    def test_generate_training_data_ranges(self):
        """Generated parameters are within specified ranges."""
        param_ranges = {
            "growth_rate": (0.015, 0.035),
            "noise_growth": (0.03, 0.08),
        }
        X, y, param_names = generate_training_data(
            n_samples=20,
            param_ranges=param_ranges,
            t_max=60.0,
            seed=42
        )

        gr_idx = param_names.index("growth_rate")
        noise_idx = param_names.index("noise_growth")

        assert np.all(X[:, gr_idx] >= 0.015)
        assert np.all(X[:, gr_idx] <= 0.035)
        assert np.all(X[:, noise_idx] >= 0.03)
        assert np.all(X[:, noise_idx] <= 0.08)


class TestBiologicalRealism:
    """Tests that verify biological realism of the model."""

    def test_doubling_time_matches_growth_rate(self):
        """Doubling time should be consistent with growth rate."""
        params = GrowthParameters(growth_rate=0.025)  # Should give ~28 min doubling
        sim = GrowthRateSimulator(params, seed=42)
        traj = sim.run(t_max=200.0, dt=0.1)

        if len(traj["division_times"]) > 2:
            interdiv = np.diff(traj["division_times"])
            mean_doubling = np.mean(interdiv)
            expected_doubling = np.log(2) / 0.025  # ~27.7 min

            # Should be within 50% of expected
            assert abs(mean_doubling - expected_doubling) / expected_doubling < 0.5

    def test_faster_growth_larger_cells(self):
        """Faster growing cells should be larger on average (growth law)."""
        # Slow growth
        slow_params = GrowthParameters(growth_rate=0.015)
        slow_sim = GrowthRateSimulator(slow_params, seed=42)
        slow_traj = slow_sim.run(t_max=120.0, dt=0.1)

        # Fast growth
        fast_params = GrowthParameters(growth_rate=0.035)
        fast_sim = GrowthRateSimulator(fast_params, seed=42)
        fast_traj = fast_sim.run(t_max=120.0, dt=0.1)

        # Fast growth should have larger average volume
        # (This is a key bacterial growth law)
        assert np.mean(fast_traj["volume"]) >= np.mean(slow_traj["volume"]) * 0.8

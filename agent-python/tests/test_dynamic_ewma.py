"""Unit tests for DynamicEWMAStrategy.

Run with:
    cd agent-python
    python -m pytest tests/test_dynamic_ewma.py -v

These tests mock external dependencies (MinIO, ENV) so they can run
without a live cluster.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# --- Mock external dependencies before any orchestration imports -----------
_mock_minio = MagicMock()
sys.modules.setdefault("minio", _mock_minio)
sys.modules.setdefault("minio.deleteobjects", MagicMock())

os.environ.setdefault("ENV", "dynamic_ewma,10,100")

from orchestration.parameters import Parameters
from orchestration.workload_state import WorkloadState
from orchestration.checkpoint import Checkpoint
from orchestration.strategies.dynamic_ewma import (
    DynamicEWMAStrategy,
    DEFAULT_MAX_POOL_SIZE,
    DEFAULT_MIN_POOL_SIZE,
    DEFAULT_BASE_POOL_SIZE,
    DEFAULT_ALPHA_FAST,
    DEFAULT_ALPHA_SLOW,
    DEFAULT_STABLE_THRESHOLD,
    DEFAULT_SPIKE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workload(eviction: int = 10, max_requests: int = 100) -> Parameters:
    return Parameters(eviction=eviction, max_requests=max_requests)


def _make_state(workload: Parameters, request_number: int, latencies=None) -> WorkloadState:
    state = WorkloadState(workload, request_number)
    state.latencies = latencies or []
    return state


def _make_checkpoint(workload: Parameters, request_number: int) -> MagicMock:
    chkpt = MagicMock(spec=Checkpoint)
    mock_state = MagicMock()
    mock_state.request_number = request_number
    chkpt.state = mock_state
    chkpt.delete = MagicMock()
    return chkpt


# ===========================================================================
# Test cases
# ===========================================================================


class TestInitialization(unittest.TestCase):
    """Verify __init__ sets all EWMA-specific parameters correctly."""

    def test_default_parameters(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(workload, [])

        self.assertEqual(strat.max_pool_size, DEFAULT_MAX_POOL_SIZE)
        self.assertEqual(strat.min_pool_size, DEFAULT_MIN_POOL_SIZE)
        self.assertEqual(strat.base_pool_size, DEFAULT_BASE_POOL_SIZE)
        self.assertAlmostEqual(strat.alpha_fast, DEFAULT_ALPHA_FAST)
        self.assertAlmostEqual(strat.alpha_slow, DEFAULT_ALPHA_SLOW)
        self.assertAlmostEqual(strat.stable_threshold, DEFAULT_STABLE_THRESHOLD)
        self.assertAlmostEqual(strat.spike_threshold, DEFAULT_SPIKE_THRESHOLD)
        self.assertEqual(strat.effective_capacity, DEFAULT_BASE_POOL_SIZE)
        self.assertEqual(strat._n_observations, 0)
        self.assertAlmostEqual(strat._ewma_mean, 0.0)
        self.assertAlmostEqual(strat._ewma_dev_fast, 0.0)
        self.assertAlmostEqual(strat._ewma_dev_slow, 0.0)

    def test_custom_parameters(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(
            workload, [],
            max_pool_size=30,
            min_pool_size=5,
            base_pool_size=10,
            alpha_fast=0.5,
            alpha_slow=0.01,
            stable_threshold=0.3,
            spike_threshold=3.0,
        )
        self.assertEqual(strat.max_pool_size, 30)
        self.assertEqual(strat.min_pool_size, 5)
        self.assertEqual(strat.effective_capacity, 10)
        self.assertAlmostEqual(strat.alpha_fast, 0.5)
        self.assertAlmostEqual(strat.alpha_slow, 0.01)

    def test_weights_array_length(self):
        workload = _make_workload(max_requests=50)
        strat = DynamicEWMAStrategy(workload, [])
        self.assertEqual(len(strat.weights), 50)
        self.assertTrue(np.all(strat.weights == 0))


class TestProperties(unittest.TestCase):
    """Test name and strategy properties."""

    def test_name_includes_pool_bounds(self):
        strat = DynamicEWMAStrategy(_make_workload(), [], max_pool_size=15, min_pool_size=3)
        self.assertIn("Max15", strat.name)
        self.assertIn("Min3", strat.name)
        self.assertIn("DynamicEWMA", strat.name)

    def test_strategy_string(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        self.assertEqual(strat.strategy, "DynamicEWMA")


class TestEWMAUpdate(unittest.TestCase):
    """Test the dual-rate EWMA update logic."""

    def setUp(self):
        self.strat = DynamicEWMAStrategy(_make_workload(), [])

    def test_first_observation_seeds_mean(self):
        self.strat._update_ewma(100.0)
        self.assertAlmostEqual(self.strat._ewma_mean, 100.0)
        self.assertAlmostEqual(self.strat._ewma_dev_fast, 0.0)
        self.assertAlmostEqual(self.strat._ewma_dev_slow, 0.0)
        self.assertEqual(self.strat._n_observations, 1)

    def test_second_observation_updates_deviation(self):
        self.strat._update_ewma(100.0)
        self.strat._update_ewma(200.0)
        # After second observation, deviations should be > 0
        self.assertGreater(self.strat._ewma_dev_fast, 0.0)
        self.assertGreater(self.strat._ewma_dev_slow, 0.0)
        self.assertEqual(self.strat._n_observations, 2)

    def test_stable_sequence_low_deviation(self):
        """Feeding identical values should keep deviations near zero."""
        for _ in range(50):
            self.strat._update_ewma(100.0)
        # After many identical values, deviation should be very small
        self.assertLess(self.strat._ewma_dev_fast, 1.0)

    def test_volatile_sequence_high_deviation(self):
        """Alternating extreme values should produce high deviations."""
        for i in range(50):
            val = 50 if i % 2 == 0 else 5000
            self.strat._update_ewma(val)
        self.assertGreater(self.strat._ewma_dev_fast, 100.0)

    def test_fast_reacts_quicker_than_slow(self):
        """After a sudden spike, fast EWMA should respond more than slow."""
        # Feed stable values
        for _ in range(30):
            self.strat._update_ewma(100.0)

        fast_before = self.strat._ewma_dev_fast
        slow_before = self.strat._ewma_dev_slow

        # Inject a big spike
        self.strat._update_ewma(10000.0)

        fast_jump = self.strat._ewma_dev_fast - fast_before
        slow_jump = self.strat._ewma_dev_slow - slow_before

        self.assertGreater(fast_jump, slow_jump)


class TestVolatilityRatio(unittest.TestCase):
    """Test the volatility ratio computation."""

    def test_neutral_with_no_data(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        self.assertAlmostEqual(strat._compute_volatility_ratio(), 1.0)

    def test_neutral_with_one_observation(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        strat._update_ewma(100.0)
        self.assertAlmostEqual(strat._compute_volatility_ratio(), 1.0)

    def test_ratio_near_one_for_stable_data(self):
        """With constant data, both EWMAs converge and ratio → 1.0"""
        strat = DynamicEWMAStrategy(_make_workload(), [])
        for _ in range(100):
            strat._update_ewma(100.0)
        ratio = strat._compute_volatility_ratio()
        # After many identical observations, ratio should be close to 1.0
        # (both fast and slow converge to the same near-zero deviation)
        # Actually with identical values, both should be ~0, so ratio would
        # be caught by the < 1e-9 guard → returns 1.0
        self.assertAlmostEqual(ratio, 1.0, places=1)

    def test_ratio_spikes_on_sudden_volatility(self):
        """After stable data, a sudden change should push ratio > 1."""
        strat = DynamicEWMAStrategy(_make_workload(), [])
        for _ in range(50):
            strat._update_ewma(100.0)

        # Inject sudden volatility
        for i in range(10):
            strat._update_ewma(50 if i % 2 == 0 else 5000)

        ratio = strat._compute_volatility_ratio()
        self.assertGreater(ratio, 1.0)


class TestUpdateEffectiveCapacity(unittest.TestCase):
    """Test pool capacity scaling from volatility ratio."""

    def _make_strategy(self, **kwargs):
        defaults = dict(
            max_pool_size=20,
            min_pool_size=2,
            base_pool_size=8,
            stable_threshold=0.5,
            spike_threshold=2.0,
        )
        defaults.update(kwargs)
        return DynamicEWMAStrategy(_make_workload(), [], **defaults)

    def test_capacity_starts_at_base(self):
        strat = self._make_strategy(base_pool_size=12)
        self.assertEqual(strat.effective_capacity, 12)

    def test_capacity_never_exceeds_max(self):
        strat = self._make_strategy(max_pool_size=10)
        # Force high ratio by manipulating internal state
        strat._ewma_dev_fast = 100.0
        strat._ewma_dev_slow = 1.0
        strat._n_observations = 50
        strat._update_effective_capacity()
        self.assertLessEqual(strat.effective_capacity, 10)

    def test_capacity_never_below_min(self):
        strat = self._make_strategy(min_pool_size=4)
        # Force low ratio
        strat._ewma_dev_fast = 0.01
        strat._ewma_dev_slow = 100.0
        strat._n_observations = 50
        strat._update_effective_capacity()
        self.assertGreaterEqual(strat.effective_capacity, 4)

    def test_high_ratio_gives_max_pool(self):
        strat = self._make_strategy()
        # ratio = 100/1 = 100 >> spike_threshold
        strat._ewma_dev_fast = 100.0
        strat._ewma_dev_slow = 1.0
        strat._n_observations = 50
        strat._update_effective_capacity()
        self.assertEqual(strat.effective_capacity, 20)

    def test_low_ratio_gives_min_pool(self):
        strat = self._make_strategy()
        # ratio = 0.1/1 = 0.1 < stable_threshold
        strat._ewma_dev_fast = 0.1
        strat._ewma_dev_slow = 1.0
        strat._n_observations = 50
        strat._update_effective_capacity()
        self.assertEqual(strat.effective_capacity, 2)

    def test_moderate_ratio_interpolates(self):
        strat = self._make_strategy()
        # ratio = 1.25 is between 0.5 and 2.0
        strat._ewma_dev_fast = 1.25
        strat._ewma_dev_slow = 1.0
        strat._n_observations = 50
        strat._update_effective_capacity()
        self.assertGreater(strat.effective_capacity, 2)
        self.assertLess(strat.effective_capacity, 20)


class TestOnRequest(unittest.TestCase):
    """Test on_request: weight update + EWMA pool sizing."""

    def test_first_request_sets_weight(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(workload, [])
        state = _make_state(workload, request_number=1, latencies=[500])
        strat.on_request(state)
        self.assertEqual(strat.weights[0], 500)

    def test_subsequent_request_uses_ema(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(workload, [], eps=0.5)
        state1 = _make_state(workload, request_number=1, latencies=[100])
        strat.on_request(state1)
        state2 = _make_state(workload, request_number=1, latencies=[200])
        strat.on_request(state2)
        self.assertAlmostEqual(strat.weights[0], 150.0)

    def test_ewma_state_updated_on_request(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(workload, [])
        state = _make_state(workload, request_number=1, latencies=[500])
        strat.on_request(state)
        self.assertEqual(strat._n_observations, 1)
        self.assertAlmostEqual(strat._ewma_mean, 500.0)

    def test_stable_requests_shrink_pool(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(
            workload, [], base_pool_size=8, min_pool_size=2, max_pool_size=20
        )
        # Feed many identical latencies → volatility ratio → 1.0 or caught by
        # zero-baseline guard → capacity stays moderate or shrinks
        for i in range(1, 50):
            state = _make_state(workload, request_number=i, latencies=[100])
            strat.on_request(state)
        # After many stable requests, capacity should not be at max
        self.assertLess(strat.effective_capacity, 20)


class TestInherentHighVariance(unittest.TestCase):
    """Core test: functions with inherently high but STABLE variance
    should NOT be penalised (ratio stays near 1.0)."""

    def test_consistently_high_variance_keeps_moderate_pool(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(
            workload, [],
            base_pool_size=8, min_pool_size=2, max_pool_size=20,
            alpha_fast=0.30, alpha_slow=0.05,
            stable_threshold=0.5, spike_threshold=2.0,
        )

        # Feed 200 requests with consistently high variance (alternating)
        # Both fast and slow EWMAs should converge, ratio → ~1.0
        for i in range(1, 200):
            lat = 50 if i % 2 == 0 else 5000
            state = _make_state(workload, request_number=min(i, 99), latencies=[lat])
            strat.on_request(state)

        ratio = strat._compute_volatility_ratio()
        # Ratio should be close to 1.0 since variance is consistent
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.5)
        # Pool should NOT be at max
        self.assertLess(strat.effective_capacity, strat.max_pool_size)


class TestPrunePool(unittest.TestCase):
    """Test that _prune_pool evicts correctly."""

    def test_prune_deletes_excess_checkpoints(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(10)]
        original_pool = list(pool)

        strat = DynamicEWMAStrategy(workload, pool, max_pool_size=20, min_pool_size=2, base_pool_size=8)
        strat._effective_capacity = 8

        for i in range(workload.max_requests):
            strat.weights[i] = i + 1

        strat._prune_pool()

        self.assertLess(len(strat.pool), len(original_pool))
        self.assertLessEqual(len(strat.pool), 8)
        kept = set(strat.pool)
        removed = [c for c in original_pool if c not in kept]
        self.assertGreater(len(removed), 0)
        for chkpt in removed:
            chkpt.delete.assert_called()

    def test_prune_not_triggered_under_capacity(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(3)]
        strat = DynamicEWMAStrategy(workload, pool, base_pool_size=10)
        result = strat.checkpoint_to_use()
        self.assertEqual(len(strat.pool), 3)


class TestCheckpointToUse(unittest.TestCase):
    """Test checkpoint_to_use returns a valid checkpoint or None."""

    def test_returns_from_pool_or_none(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, 5)]
        strat = DynamicEWMAStrategy(workload, pool)
        results = set()
        for _ in range(50):
            results.add(strat.checkpoint_to_use())
        self.assertTrue(results.issubset({pool[0], None}))

    def test_empty_pool_returns_none(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        self.assertIsNone(strat.checkpoint_to_use())


class TestReset(unittest.TestCase):
    """Test that reset() clears all mutable state."""

    def test_reset_clears_pool_and_ewma(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(5)]
        strat = DynamicEWMAStrategy(workload, pool, base_pool_size=8)

        strat._ewma_mean = 500.0
        strat._ewma_dev_fast = 50.0
        strat._ewma_dev_slow = 30.0
        strat._n_observations = 100
        strat._effective_capacity = 15

        strat.reset()

        self.assertEqual(len(strat.pool), 0)
        self.assertAlmostEqual(strat._ewma_mean, 0.0)
        self.assertAlmostEqual(strat._ewma_dev_fast, 0.0)
        self.assertAlmostEqual(strat._ewma_dev_slow, 0.0)
        self.assertEqual(strat._n_observations, 0)
        self.assertEqual(strat.effective_capacity, 8)  # reset to base
        self.assertTrue(np.all(strat.weights == 0))

    def test_reset_calls_delete_on_checkpoints(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(3)]
        strat = DynamicEWMAStrategy(workload, pool)
        strat.reset()
        for chkpt in pool:
            chkpt.delete.assert_called_once()


class TestSerialization(unittest.TestCase):
    """Test extra_state and round-trip serialization."""

    def test_extra_state_keys(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        state = strat.extra_state

        expected_keys = {
            "max_pool_size", "min_pool_size", "base_pool_size",
            "p", "gamma", "weights", "eps",
            "alpha_fast", "alpha_slow",
            "stable_threshold", "spike_threshold",
            "ewma_mean", "ewma_dev_fast", "ewma_dev_slow",
            "n_observations", "effective_capacity",
        }
        self.assertEqual(set(state.keys()), expected_keys)

    def test_serialize_roundtrip(self):
        strat = DynamicEWMAStrategy(_make_workload(), [])
        strat._ewma_mean = 123.4
        strat._ewma_dev_fast = 45.6
        strat._ewma_dev_slow = 12.3
        strat._n_observations = 50
        strat._effective_capacity = 12

        payload = strat.serialize()
        obj = json.loads(payload)

        self.assertEqual(obj["strategy"], "DynamicEWMA")
        self.assertAlmostEqual(obj["ewma_mean"], 123.4)
        self.assertAlmostEqual(obj["ewma_dev_fast"], 45.6)
        self.assertEqual(obj["n_observations"], 50)
        self.assertEqual(obj["effective_capacity"], 12)

    def test_deserialization_restores_state(self):
        from orchestration.utils import cr_deserialize

        strat = DynamicEWMAStrategy(_make_workload(), [])
        strat._ewma_mean = 200.0
        strat._ewma_dev_fast = 30.0
        strat._ewma_dev_slow = 10.0
        strat._n_observations = 75
        strat._effective_capacity = 7
        strat.weights[0] = 42

        payload = strat.serialize()
        mock_client = MagicMock()
        restored = cr_deserialize(payload, mock_client)

        self.assertIsInstance(restored, DynamicEWMAStrategy)
        self.assertEqual(restored.max_pool_size, strat.max_pool_size)
        self.assertAlmostEqual(restored._ewma_mean, 200.0)
        self.assertAlmostEqual(restored._ewma_dev_fast, 30.0)
        self.assertAlmostEqual(restored._ewma_dev_slow, 10.0)
        self.assertEqual(restored._n_observations, 75)
        self.assertEqual(restored._effective_capacity, 7)
        self.assertAlmostEqual(restored.weights[0], 42)


class TestCapacityTransitions(unittest.TestCase):
    """Simulate workloads that transition between stable and volatile."""

    def test_spike_after_stable_increases_capacity(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(
            workload, [], base_pool_size=8, min_pool_size=2, max_pool_size=20
        )

        # Phase 1: stable
        for i in range(1, 60):
            state = _make_state(workload, request_number=min(i, 99), latencies=[100])
            strat.on_request(state)
        cap_stable = strat.effective_capacity

        # Phase 2: sudden volatility spike
        for i in range(60, 80):
            lat = 50 if i % 2 == 0 else 5000
            state = _make_state(workload, request_number=min(i, 99), latencies=[lat])
            strat.on_request(state)
        cap_volatile = strat.effective_capacity

        self.assertGreater(cap_volatile, cap_stable)

    def test_stabilising_after_spike_decreases_capacity(self):
        workload = _make_workload()
        strat = DynamicEWMAStrategy(
            workload, [], base_pool_size=10, min_pool_size=2, max_pool_size=20
        )

        # Phase 1: volatile
        for i in range(1, 60):
            lat = 50 if i % 2 == 0 else 5000
            state = _make_state(workload, request_number=min(i, 99), latencies=[lat])
            strat.on_request(state)
        cap_volatile = strat.effective_capacity

        # Phase 2: stabilise
        for i in range(60, 150):
            state = _make_state(workload, request_number=min(i, 99), latencies=[100])
            strat.on_request(state)
        cap_stable = strat.effective_capacity

        self.assertLess(cap_stable, cap_volatile)


if __name__ == "__main__":
    unittest.main()

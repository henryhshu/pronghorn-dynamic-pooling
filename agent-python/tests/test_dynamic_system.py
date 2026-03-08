"""Unit tests for DynamicSystemStrategy.

Run with:
    cd agent-python
    python -m pytest tests/test_dynamic_system.py -v

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
# minio is not available in the local test environment, so we stub it out at
# the sys.modules level.  The strategy logic never calls MinIO directly.
_mock_minio = MagicMock()
sys.modules.setdefault("minio", _mock_minio)
sys.modules.setdefault("minio.deleteobjects", MagicMock())

# The Parameters module reads ENV at import time, so we must patch it before
# any orchestration imports.
os.environ.setdefault("ENV", "dynamic_system,10,100")

from orchestration.parameters import Parameters
from orchestration.workload_state import WorkloadState
from orchestration.checkpoint import Checkpoint
from orchestration.strategies.dynamic_system import (
    DynamicSystemStrategy,
    DEFAULT_MAX_POOL_SIZE,
    DEFAULT_MIN_POOL_SIZE,
    DEFAULT_BASE_POOL_SIZE,
    DEFAULT_LOW_VARIANCE_THRESHOLD,
    DEFAULT_HIGH_VARIANCE_THRESHOLD,
    DEFAULT_VARIANCE_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workload(eviction: int = 10, max_requests: int = 100) -> Parameters:
    """Create a Parameters object with explicit values."""
    return Parameters(eviction=eviction, max_requests=max_requests)


def _make_state(workload: Parameters, request_number: int, latencies=None) -> WorkloadState:
    """Create a WorkloadState at a given request number with supplied latencies."""
    state = WorkloadState(workload, request_number)
    state.latencies = latencies or []
    return state


def _make_checkpoint(workload: Parameters, request_number: int) -> MagicMock:
    """Return a mock Checkpoint at the given request number."""
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
    """Verify that __init__ sets all parameters correctly."""

    def test_default_parameters(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(workload, [])

        self.assertEqual(strat.max_pool_size, DEFAULT_MAX_POOL_SIZE)
        self.assertEqual(strat.min_pool_size, DEFAULT_MIN_POOL_SIZE)
        self.assertEqual(strat.base_pool_size, DEFAULT_BASE_POOL_SIZE)
        self.assertEqual(strat.low_var_thresh, DEFAULT_LOW_VARIANCE_THRESHOLD)
        self.assertEqual(strat.high_var_thresh, DEFAULT_HIGH_VARIANCE_THRESHOLD)
        self.assertEqual(strat.variance_window, DEFAULT_VARIANCE_WINDOW)
        self.assertEqual(strat.effective_capacity, DEFAULT_BASE_POOL_SIZE)
        self.assertEqual(len(strat._recent_latencies), 0)

    def test_custom_parameters(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(
            workload, [],
            max_pool_size=30,
            min_pool_size=5,
            base_pool_size=10,
            low_var_thresh=0.05,
            high_var_thresh=2.0,
            variance_window=100,
        )
        self.assertEqual(strat.max_pool_size, 30)
        self.assertEqual(strat.min_pool_size, 5)
        self.assertEqual(strat.base_pool_size, 10)
        self.assertEqual(strat.effective_capacity, 10)

    def test_weights_array_length(self):
        workload = _make_workload(max_requests=50)
        strat = DynamicSystemStrategy(workload, [])
        self.assertEqual(len(strat.weights), 50)
        self.assertTrue(np.all(strat.weights == 0))


class TestProperties(unittest.TestCase):
    """Test name and strategy properties."""

    def test_name_includes_pool_bounds(self):
        strat = DynamicSystemStrategy(_make_workload(), [], max_pool_size=15, min_pool_size=3)
        self.assertIn("Max15", strat.name)
        self.assertIn("Min3", strat.name)

    def test_strategy_string(self):
        strat = DynamicSystemStrategy(_make_workload(), [])
        self.assertEqual(strat.strategy, "DynamicSystem")


class TestComputeCV(unittest.TestCase):
    """Test the coefficient-of-variation computation."""

    def setUp(self):
        self.strat = DynamicSystemStrategy(_make_workload(), [])

    def test_cv_with_no_data(self):
        self.assertAlmostEqual(self.strat._compute_cv(), 0.0)

    def test_cv_with_one_datapoint(self):
        self.strat._recent_latencies = [100]
        self.assertAlmostEqual(self.strat._compute_cv(), 0.0)

    def test_cv_with_identical_values(self):
        """All-identical latencies → stddev=0 → CV=0."""
        self.strat._recent_latencies = [100] * 20
        self.assertAlmostEqual(self.strat._compute_cv(), 0.0)

    def test_cv_with_known_values(self):
        """Hand-compute CV for a small dataset and compare."""
        data = [100, 200, 300]
        self.strat._recent_latencies = data
        arr = np.array(data)
        expected_cv = float(np.std(arr) / np.mean(arr))
        self.assertAlmostEqual(self.strat._compute_cv(), expected_cv, places=6)

    def test_cv_respects_variance_window(self):
        """Only the last `variance_window` entries should count."""
        self.strat.variance_window = 3
        # Pad with high-variance junk, then three identical values at the tail
        self.strat._recent_latencies = [1, 1000, 1, 1000, 100, 100, 100]
        # CV should be computed only on [100, 100, 100]
        self.assertAlmostEqual(self.strat._compute_cv(), 0.0)

    def test_cv_with_zero_mean(self):
        """Zero-mean latencies should return 0.0 (guard against division by zero)."""
        self.strat._recent_latencies = [0, 0, 0]
        self.assertAlmostEqual(self.strat._compute_cv(), 0.0)


class TestUpdateEffectiveCapacity(unittest.TestCase):
    """Test that pool capacity is scaled correctly based on CV."""

    def _make_strategy(self, **kwargs):
        defaults = dict(
            max_pool_size=20,
            min_pool_size=2,
            base_pool_size=8,
            low_var_thresh=0.10,
            high_var_thresh=1.0,
        )
        defaults.update(kwargs)
        return DynamicSystemStrategy(_make_workload(), [], **defaults)

    def test_low_cv_gives_min_pool(self):
        strat = self._make_strategy()
        strat._recent_latencies = [100] * 20  # CV ≈ 0
        strat._update_effective_capacity()
        self.assertEqual(strat.effective_capacity, 2)

    def test_high_cv_gives_max_pool(self):
        strat = self._make_strategy()
        # Generate data with CV that clearly exceeds 1.0
        # A mix of zeros and large values guarantees CV > 1
        strat._recent_latencies = [1] * 10 + [100000] * 10
        strat._update_effective_capacity()
        self.assertEqual(strat.effective_capacity, 20)

    def test_moderate_cv_interpolates(self):
        strat = self._make_strategy()
        # Inject latencies that produce a CV of ~0.55 (midpoint of [0.1, 1.0])
        # For a uniform-ish distribution between a and b:
        # CV = (b-a) / (sqrt(12) * (a+b)/2) — but let's just force it
        strat._recent_latencies = [100, 200, 100, 200, 100, 200]
        cv = strat._compute_cv()
        # Verify within the interpolation range
        self.assertGreater(cv, 0.10)
        self.assertLess(cv, 1.0)
        strat._update_effective_capacity()
        self.assertGreater(strat.effective_capacity, 2)
        self.assertLess(strat.effective_capacity, 20)

    def test_capacity_never_exceeds_max(self):
        strat = self._make_strategy(max_pool_size=10)
        strat._recent_latencies = [1, 10000] * 30
        strat._update_effective_capacity()
        self.assertLessEqual(strat.effective_capacity, 10)

    def test_capacity_never_below_min(self):
        strat = self._make_strategy(min_pool_size=4)
        strat._recent_latencies = [100] * 20
        strat._update_effective_capacity()
        self.assertGreaterEqual(strat.effective_capacity, 4)

    def test_capacity_starts_at_base(self):
        strat = self._make_strategy(base_pool_size=12)
        self.assertEqual(strat.effective_capacity, 12)


class TestOnRequest(unittest.TestCase):
    """Test on_request: weight update + dynamic pool sizing."""

    def test_first_request_sets_weight(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(workload, [])
        state = _make_state(workload, request_number=1, latencies=[500])

        strat.on_request(state)

        self.assertEqual(strat.weights[0], 500)

    def test_subsequent_request_uses_ema(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(workload, [], eps=0.5)
        # First request
        state1 = _make_state(workload, request_number=1, latencies=[100])
        strat.on_request(state1)
        # Second request at the same position
        state2 = _make_state(workload, request_number=1, latencies=[200])
        strat.on_request(state2)
        # EMA: 0.5 * 200 + 0.5 * 100 = 150
        self.assertAlmostEqual(strat.weights[0], 150.0)

    def test_recent_latencies_tracked(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(workload, [], variance_window=5)

        for i in range(1, 8):
            state = _make_state(workload, request_number=i, latencies=[i * 100])
            strat.on_request(state)

        # Window is 5, so only the last 5 should remain
        self.assertLessEqual(len(strat._recent_latencies), 5)

    def test_on_request_triggers_capacity_update(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(
            workload, [],
            base_pool_size=8,
            min_pool_size=2,
            max_pool_size=20,
        )
        # Feed many identical latencies → CV ≈ 0 → should shrink to min
        for i in range(1, 20):
            state = _make_state(workload, request_number=i, latencies=[100])
            strat.on_request(state)

        self.assertEqual(strat.effective_capacity, 2)


class TestPrunePool(unittest.TestCase):
    """Test that _prune_pool evicts correctly."""

    def test_prune_deletes_excess_checkpoints(self):
        workload = _make_workload()
        # Use 10 checkpoints with p=0.40 → keeps 4, gamma=0.10 of remaining 6 → keeps ~1
        # Total kept ≈ 5, which fits in effective_capacity=8
        pool = [_make_checkpoint(workload, i) for i in range(10)]
        original_pool = list(pool)

        strat = DynamicSystemStrategy(workload, pool, max_pool_size=20, min_pool_size=2, base_pool_size=8)
        strat._effective_capacity = 8

        # Ensure weights are non-zero so sorting works
        for i in range(workload.max_requests):
            strat.weights[i] = i + 1

        strat._prune_pool()

        self.assertLess(len(strat.pool), len(original_pool))
        self.assertLessEqual(len(strat.pool), 8)
        # Verify delete() was called on removed checkpoints
        kept = set(strat.pool)
        removed = [c for c in original_pool if c not in kept]
        self.assertGreater(len(removed), 0)
        for chkpt in removed:
            chkpt.delete.assert_called()

    def test_prune_not_triggered_under_capacity(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(3)]
        strat = DynamicSystemStrategy(workload, pool, base_pool_size=10)

        # checkpoint_to_use should NOT prune when pool < effective_capacity
        result = strat.checkpoint_to_use()
        self.assertEqual(len(strat.pool), 3)


class TestCheckpointToUse(unittest.TestCase):
    """Test checkpoint_to_use returns a valid checkpoint or None."""

    def test_returns_from_pool_or_none(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, 5)]
        strat = DynamicSystemStrategy(workload, pool)

        results = set()
        for _ in range(50):
            result = strat.checkpoint_to_use()
            results.add(result)

        # Should return either the checkpoint or None
        possible = {pool[0], None}
        self.assertTrue(results.issubset(possible))

    def test_empty_pool_returns_none(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(workload, [])

        # With an empty pool, expanded_pool = [None], so must return None
        result = strat.checkpoint_to_use()
        self.assertIsNone(result)


class TestReset(unittest.TestCase):
    """Test that reset() clears all mutable state."""

    def test_reset_clears_pool(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(5)]
        strat = DynamicSystemStrategy(workload, pool, base_pool_size=8)

        strat._recent_latencies = [100, 200, 300]
        strat._effective_capacity = 15

        strat.reset()

        self.assertEqual(len(strat.pool), 0)
        self.assertEqual(len(strat._recent_latencies), 0)
        self.assertEqual(strat.effective_capacity, 8)  # reset to base
        self.assertTrue(np.all(strat.weights == 0))

    def test_reset_calls_delete_on_checkpoints(self):
        workload = _make_workload()
        pool = [_make_checkpoint(workload, i) for i in range(3)]
        strat = DynamicSystemStrategy(workload, pool)

        strat.reset()

        for chkpt in pool:
            chkpt.delete.assert_called_once()


class TestSerialization(unittest.TestCase):
    """Test extra_state and round-trip serialization."""

    def test_extra_state_keys(self):
        strat = DynamicSystemStrategy(_make_workload(), [])
        state = strat.extra_state

        expected_keys = {
            "max_pool_size", "min_pool_size", "base_pool_size",
            "p", "gamma", "weights", "eps",
            "low_var_thresh", "high_var_thresh", "variance_window",
            "recent_latencies", "effective_capacity",
        }
        self.assertEqual(set(state.keys()), expected_keys)

    def test_serialize_roundtrip(self):
        """Verify that serialize() produces valid JSON with all fields."""
        strat = DynamicSystemStrategy(_make_workload(), [])
        strat._recent_latencies = [100, 200, 300]
        strat._effective_capacity = 12

        payload = strat.serialize()
        obj = json.loads(payload)

        self.assertEqual(obj["strategy"], "DynamicSystem")
        self.assertEqual(obj["max_pool_size"], DEFAULT_MAX_POOL_SIZE)
        self.assertEqual(obj["effective_capacity"], 12)
        self.assertEqual(obj["recent_latencies"], [100, 200, 300])

    def test_deserialization_restores_state(self):
        """Verify cr_deserialize restores a DynamicSystemStrategy."""
        from orchestration.utils import cr_deserialize

        strat = DynamicSystemStrategy(_make_workload(), [])
        strat._recent_latencies = [50, 150, 250]
        strat._effective_capacity = 7
        strat.weights[0] = 42

        payload = strat.serialize()

        mock_client = MagicMock()
        restored = cr_deserialize(payload, mock_client)

        self.assertIsInstance(restored, DynamicSystemStrategy)
        self.assertEqual(restored.max_pool_size, strat.max_pool_size)
        self.assertEqual(restored.min_pool_size, strat.min_pool_size)
        self.assertEqual(restored._effective_capacity, 7)
        self.assertEqual(restored._recent_latencies, [50, 150, 250])
        self.assertAlmostEqual(restored.weights[0], 42)


class TestVarianceWindowBoundary(unittest.TestCase):
    """Edge cases around the variance window."""

    def test_exactly_window_size(self):
        strat = DynamicSystemStrategy(_make_workload(), [], variance_window=5)
        strat._recent_latencies = [10, 20, 30, 40, 50]
        cv = strat._compute_cv()
        arr = np.array([10, 20, 30, 40, 50])
        expected = float(np.std(arr) / np.mean(arr))
        self.assertAlmostEqual(cv, expected, places=6)

    def test_more_than_window_uses_tail(self):
        strat = DynamicSystemStrategy(_make_workload(), [], variance_window=3)
        strat._recent_latencies = [999, 999, 999, 100, 100, 100]
        # Should only use the last 3 values [100, 100, 100] → CV = 0
        self.assertAlmostEqual(strat._compute_cv(), 0.0)


class TestCapacityTransitions(unittest.TestCase):
    """Simulate a workload that transitions from stable to volatile."""

    def test_capacity_increases_with_variance(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(
            workload, [],
            base_pool_size=5,
            min_pool_size=2,
            max_pool_size=20,
            variance_window=10,
        )

        # Phase 1: stable latencies → should shrink toward min
        for i in range(1, 15):
            state = _make_state(workload, request_number=i, latencies=[100])
            strat.on_request(state)
        cap_stable = strat.effective_capacity

        # Phase 2: highly variable latencies → should grow toward max
        for i in range(15, 30):
            lat = 50 if i % 2 == 0 else 5000
            state = _make_state(workload, request_number=min(i, 99), latencies=[lat])
            strat.on_request(state)
        cap_volatile = strat.effective_capacity

        self.assertGreater(cap_volatile, cap_stable)

    def test_capacity_decreases_when_stabilising(self):
        workload = _make_workload()
        strat = DynamicSystemStrategy(
            workload, [],
            base_pool_size=10,
            min_pool_size=2,
            max_pool_size=20,
            variance_window=10,
        )

        # Phase 1: volatile
        for i in range(1, 15):
            lat = 50 if i % 2 == 0 else 5000
            state = _make_state(workload, request_number=min(i, 99), latencies=[lat])
            strat.on_request(state)
        cap_volatile = strat.effective_capacity

        # Phase 2: stabilise
        for i in range(15, 30):
            state = _make_state(workload, request_number=min(i, 99), latencies=[100])
            strat.on_request(state)
        cap_stable = strat.effective_capacity

        self.assertLess(cap_stable, cap_volatile)


if __name__ == "__main__":
    unittest.main()

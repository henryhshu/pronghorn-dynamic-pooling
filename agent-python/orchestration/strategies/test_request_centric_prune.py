import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Parameters module reads ENV at import time
os.environ.setdefault("ENV", "test,test,5")

# Mock minio so we can run without minio installed (smoke test only)
sys.modules["minio"] = MagicMock()
sys.modules["minio.deleteobjects"] = MagicMock()

# So tests run from repo root or agent-python; orchestration is in agent-python
_agent_python = Path(__file__).resolve().parents[2]
if str(_agent_python) not in sys.path:
    sys.path.insert(0, str(_agent_python))

import numpy as np

from orchestration.checkpoint import Checkpoint
from orchestration.workload_state import WorkloadState
from orchestration.parameters import Parameters
from orchestration.strategies.pruning_request_centric import PruningRequestCentricStrategy


def make_checkpoint(latencies, req_num, avg_response_time=None, var_response_time=None):
    ws = WorkloadState(Parameters(eviction=5, max_requests=10), req_num)
    ws.latencies = latencies
    return Checkpoint(
        ws, path=f"chkpt_{req_num}",
        avg_response_time=avg_response_time, var_response_time=var_response_time,
    )


def test_score_penalizes_high_avg_low_var():
    """Low avg + high var should score higher than high avg + low var."""
    workload = Parameters(eviction=5, max_requests=10)
    good = make_checkpoint([10, 12, 11], 1)   # low avg, low var
    bad = make_checkpoint([100, 102, 101], 2)  # high avg, low var
    pool = [good, bad]
    strategy = PruningRequestCentricStrategy(workload, pool.copy(), max_capacity=10)
    assert strategy._score_for_checkpoint(good) > strategy._score_for_checkpoint(bad)


def test_target_size_converged_weighted_with_capped_buffer():
    """With explicit buffer (e.g. 2): target = min_pool_size when var tiny, min_pool_size+buffer when var near threshold."""
    workload = Parameters(eviction=5, max_requests=10)
    min_size = 3
    max_cap = 14
    buffer = 2
    threshold = 100.0
    low_var_pool = [
        make_checkpoint([5, 5, 5], i) for i in range(5)
    ]
    strategy_low = PruningRequestCentricStrategy(
        workload, low_var_pool, max_capacity=max_cap,
        min_pool_size=min_size, variance_converged_threshold=threshold,
        converged_shrink_buffer=buffer,
    )
    assert strategy_low._target_size_converged() == min_size

    high_var_pool = [
        make_checkpoint([0], 1, avg_response_time=10, var_response_time=99),
        make_checkpoint([0], 2, avg_response_time=15, var_response_time=80),
        make_checkpoint([0], 3, avg_response_time=20, var_response_time=50),
        make_checkpoint([0], 4, avg_response_time=25, var_response_time=30),
        make_checkpoint([0], 5, avg_response_time=30, var_response_time=10),
    ]
    strategy_high = PruningRequestCentricStrategy(
        workload, high_var_pool, max_capacity=max_cap,
        min_pool_size=min_size, variance_converged_threshold=threshold,
        converged_shrink_buffer=buffer,
    )
    assert strategy_high._target_size_converged() == min_size + buffer


def test_target_size_converged_weighted_default_buffer():
    """With default buffer=None, buffer scales with headroom: var tiny -> min_pool_size, var at threshold -> max_capacity."""
    workload = Parameters(eviction=5, max_requests=10)
    min_size = 3
    max_cap = 14
    threshold = 100.0
    headroom = max_cap - min_size  # 11

    low_var_pool = [make_checkpoint([1, 1, 1], i) for i in range(7)]
    strategy_low = PruningRequestCentricStrategy(
        workload, low_var_pool, max_capacity=max_cap,
        min_pool_size=min_size, variance_converged_threshold=threshold,
        # converged_shrink_buffer=None is default
    )
    assert strategy_low._target_size_converged() == min_size

    # Pool size >= max_cap so target is not capped by len(pool)
    var_at_threshold_pool = [
        make_checkpoint([0], i, avg_response_time=10 * i, var_response_time=threshold)
        for i in range(1, max_cap + 1)
    ]
    strategy_at_threshold = PruningRequestCentricStrategy(
        workload, var_at_threshold_pool, max_capacity=max_cap,
        min_pool_size=min_size, variance_converged_threshold=threshold,
    )
    assert strategy_at_threshold._target_size_converged() == max_cap


def test_target_size_converged_weighted_scales_with_ratio():
    """Target size scales linearly with ratio (how close var is to threshold) when using default buffer."""
    workload = Parameters(eviction=5, max_requests=10)
    min_size = 2
    max_cap = 10
    threshold = 100.0
    headroom = max_cap - min_size  # 8

    # ratio = 0.5 -> extra = round(0.5 * 8) = 4 -> target = 6
    mid_var_pool = [
        make_checkpoint([0], i, avg_response_time=5 * i, var_response_time=50.0) for i in range(1, 9)
    ]
    strategy = PruningRequestCentricStrategy(
        workload, mid_var_pool, max_capacity=max_cap,
        min_pool_size=min_size, variance_converged_threshold=threshold,
    )
    assert strategy._target_size_converged() == min_size + 4  # 6


def test_is_variance_converged():
    """Converged when max var in pool <= threshold."""
    workload = Parameters(eviction=5, max_requests=10)
    low_var = [make_checkpoint([1, 1, 1], i) for i in range(3)]
    strategy = PruningRequestCentricStrategy(
        workload, low_var, variance_converged_threshold=100.0,
    )
    assert strategy._is_variance_converged() is True

    high_var = [make_checkpoint([1, 50, 100], i) for i in range(3)]  # var large
    strategy2 = PruningRequestCentricStrategy(
        workload, high_var, variance_converged_threshold=100.0,
    )
    assert strategy2._is_variance_converged() is False


def test_shrink_pool_converged_reduces_to_target():
    """Converged shrink should reduce pool to weighted target, keeping top by score."""
    workload = Parameters(eviction=5, max_requests=10)
    pool = [
        make_checkpoint([5, 5, 5], 1),
        make_checkpoint([6, 6, 6], 2),
        make_checkpoint([7, 7, 7], 3),
        make_checkpoint([8, 8, 8], 4),
        make_checkpoint([9, 9, 9], 5),
    ]
    strategy = PruningRequestCentricStrategy(
        workload, pool, max_capacity=14,
        min_pool_size=2, variance_converged_threshold=100.0,
        converged_shrink_buffer=1,
    )
    with patch.object(Checkpoint, "delete"):
        strategy._shrink_pool_converged()
    # Very low var -> target = min_pool_size = 2
    assert len(strategy.pool) == 2
    assert strategy._score_for_checkpoint(strategy.pool[0]) >= strategy._score_for_checkpoint(strategy.pool[1])


# --- Original script-style run (optional) ---
if __name__ == "__main__":
    test_score_penalizes_high_avg_low_var()
    test_target_size_converged_weighted_with_capped_buffer()
    test_target_size_converged_weighted_default_buffer()
    test_target_size_converged_weighted_scales_with_ratio()
    test_is_variance_converged()
    test_shrink_pool_converged_reduces_to_target()
    print("All lightweight tests passed.")

    # Original prune demo (no assertions)
    high_avg_low_var = make_checkpoint([100, 102, 101, 99, 100], 1)
    low_avg_high_var = make_checkpoint([10, 50, 90, 130, 170], 2)
    low_avg_low_var = make_checkpoint([10, 12, 11, 9, 10], 3)
    high_avg_high_var = make_checkpoint([100, 50, 200, 150, 100], 4)
    pool = [high_avg_low_var, low_avg_high_var, low_avg_low_var, high_avg_high_var]
    workload = Parameters(eviction=5, max_requests=10)
    strategy = PruningRequestCentricStrategy(workload, pool, max_capacity=4)
    print("\nBefore pruning:")
    for c in pool:
        print(f"  {c.path}: avg={c.avg_response_time:.2f}, var={c.var_response_time:.2f}")
    with patch.object(Checkpoint, "delete"):
        strategy._prune_pool()
    print("After pruning:")
    for c in strategy.pool:
        print(f"  {c.path}: avg={c.avg_response_time:.2f}, var={c.var_response_time:.2f}")

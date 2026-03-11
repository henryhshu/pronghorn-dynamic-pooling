from typing import List, Union

from .. import Parameters, CRStrategy, Checkpoint, WorkloadState, FixedStrategy
import random
import numpy as np
import copy

DEFAULT_MAX_POOL_SIZE = 14
DEFAULT_MIN_POOL_SIZE = 2
DEFAULT_BASE_POOL_SIZE = 8

DEFAULT_P = 0.40
DEFAULT_GAMMA = 0.10

DEFAULT_PERFORMANCE_FN = lambda arr: np.mean(np.array(arr))

DEFAULT_EPSILON = 0.5
MIN_WEIGHT_EPSILON = 1  # microseconds (median latency)

# Variance thresholds for scaling the pool size.
# When the coefficient of variation (stddev / mean) of latencies exceeds
# HIGH_VARIANCE_THRESHOLD, the pool scales toward max_pool_size.
# When it drops below LOW_VARIANCE_THRESHOLD, the pool scales toward
# min_pool_size. In between, the pool size is linearly interpolated.
DEFAULT_LOW_VARIANCE_THRESHOLD = 0.10
DEFAULT_HIGH_VARIANCE_THRESHOLD = 1.0

# Number of recent latencies used when computing the running variance.
DEFAULT_VARIANCE_WINDOW = 50


class DynamicSystemStrategy(CRStrategy):
    """A CRStrategy variant with a variance-driven dynamic pool size.

    This strategy converges toward a local (system-level) optimum rather than
    a global one.  Instead of maintaining a fixed ``max_capacity`` for the
    snapshot pool, it continuously monitors the coefficient of variation (CV)
    of recent request latencies and adjusts the *effective* pool capacity
    accordingly:

    * **High CV** (> ``high_var_thresh``) → pool grows toward ``max_pool_size``
      because unpredictable latencies benefit from a wider set of snapshots.
    * **Low CV** (< ``low_var_thresh``) → pool shrinks toward ``min_pool_size``
      because the JIT compiler has stabilised and fewer snapshots suffice.
    * **In between** → the pool size is linearly interpolated.

    An absolute ``max_pool_size`` cap is always enforced so that memory usage
    cannot explode.
    """

    def __init__(
        self,
        workload: Parameters,
        pool: List[Checkpoint],
        max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
        min_pool_size: int = DEFAULT_MIN_POOL_SIZE,
        base_pool_size: int = DEFAULT_BASE_POOL_SIZE,
        p: float = DEFAULT_P,
        gamma: float = DEFAULT_GAMMA,
        performance_fn=DEFAULT_PERFORMANCE_FN,
        eps: float = DEFAULT_EPSILON,
        low_var_thresh: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
        high_var_thresh: float = DEFAULT_HIGH_VARIANCE_THRESHOLD,
        variance_window: int = DEFAULT_VARIANCE_WINDOW,
    ) -> None:
        super().__init__(workload, pool)

        # Pool size bounds
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.base_pool_size = base_pool_size

        # Checkpoint-selection tunables (same semantics as RequestCentric)
        self.p = p
        self.gamma = gamma
        self.performance_fn = performance_fn
        self.eps = eps

        # Per-request weight array (exponentially-smoothed latencies)
        self.weights = np.array([0] * workload.max_requests)

        # Variance-tracking state
        self.low_var_thresh = low_var_thresh
        self.high_var_thresh = high_var_thresh
        self.variance_window = variance_window
        self._recent_latencies: List[float] = []

        # Current effective capacity (starts at the base)
        self._effective_capacity: int = base_pool_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            f"DynamicSystem_Max{self.max_pool_size}_Min{self.min_pool_size}"
            f"_P{self.p}_Gamma{self.gamma}"
        )

    @property
    def strategy(self) -> str:
        return "DynamicSystem"

    @property
    def effective_capacity(self) -> int:
        """The current pool capacity derived from latency variance."""
        return self._effective_capacity

    # ------------------------------------------------------------------
    # Variance helpers
    # ------------------------------------------------------------------

    def _compute_cv(self) -> float:
        """Return the coefficient of variation of recent latencies.

        Returns 0.0 when there are fewer than 2 data points.
        """
        if len(self._recent_latencies) < 2:
            return 0.0
        arr = np.array(self._recent_latencies[-self.variance_window :])
        mean = np.mean(arr)
        if mean == 0:
            return 0.0
        return float(np.std(arr) / mean)

    def _update_effective_capacity(self) -> None:
        """Recalculate the effective pool capacity from the current CV."""
        cv = self._compute_cv()

        if cv <= self.low_var_thresh:
            new_capacity = self.min_pool_size
        elif cv >= self.high_var_thresh:
            new_capacity = self.max_pool_size
        else:
            # Linear interpolation between min and max based on CV
            ratio = (cv - self.low_var_thresh) / (
                self.high_var_thresh - self.low_var_thresh
            )
            new_capacity = round(
                self.min_pool_size
                + ratio * (self.max_pool_size - self.min_pool_size)
            )

        # Hard clamp to [min, max] (safety net)
        new_capacity = max(self.min_pool_size, min(new_capacity, self.max_pool_size))

        if new_capacity != self._effective_capacity:
            print(
                f"[DynamicSystem] CV={cv:.4f} → adjusting capacity "
                f"{self._effective_capacity} → {new_capacity}"
            )
        self._effective_capacity = new_capacity

    # ------------------------------------------------------------------
    # Weight helpers (same logic as RequestCentricStrategy)
    # ------------------------------------------------------------------

    def _weights_for(self, req_num, scalar=False):
        cur_slice = self.weights[
            req_num : min(req_num + self.workload.eviction, self.workload.max_requests)
        ]
        output = 1000000.0 / (cur_slice + MIN_WEIGHT_EPSILON)
        if scalar:
            output = self.performance_fn(output)
        return output

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _prune_pool(self):
        """Evict checkpoints until the pool fits within effective_capacity."""
        output = []
        # reference_size = max(self.base_pool_size, self._effective_capacity)
        reference_size = len(self.pool)

        by_performance = sorted(
            self.pool,
            key=lambda c: self._weights_for(c.state.request_number, scalar=True),
            reverse=True,
        )

        keeping_p = min(round(self.p * reference_size), self._effective_capacity)
        output += by_performance[:keeping_p]
        by_performance = by_performance[keeping_p:]

        keeping_gamma = 0
        if keeping_p < self._effective_capacity and by_performance:
            remaining_slots = self._effective_capacity - keeping_p
            keeping_gamma = min(
                round(self.gamma * reference_size),
                remaining_slots,
                len(by_performance),
            )
            output += random.choices(
                by_performance, k=keeping_gamma
            )

        output_chkpts = {chkpt for chkpt in output}
        removed = [chkpt for chkpt in self.pool if chkpt not in output_chkpts]
        for chkpt in removed:
            chkpt.delete()

        self.pool[:] = output
        print(
            f"[DynamicSystem] Evicted to {len(self.pool)} checkpoints "
            f"(top {keeping_p} by perf + {keeping_gamma} random, "
            f"effective_capacity={self._effective_capacity})"
        )
        assert len(self.pool) <= self._effective_capacity

    # ------------------------------------------------------------------
    # CRStrategy interface
    # ------------------------------------------------------------------

    def checkpoint_to_use(self) -> Checkpoint:
        if len(self.pool) > self._effective_capacity:
            self._prune_pool()

        print(f"[DynamicSystem] Selecting checkpoint (pool={len(self.pool)}, "
              f"effective_capacity={self._effective_capacity})")

        expanded_pool = self.pool + [None]
        weights = [
            self._weights_for(
                chkpt.state.request_number if chkpt is not None else 0, scalar=True
            )
            for chkpt in expanded_pool
        ]
        weights = np.array(weights)
        weights_max = np.amax(weights, keepdims=True)
        weights_shifted = np.exp(weights - weights_max)
        weights = weights_shifted / np.sum(weights_shifted, keepdims=True)
        weights = weights.tolist()
        print(list(zip(weights, expanded_pool)))
        ret_val = random.choices(expanded_pool, weights=weights, k=1)[0]
        print("[DynamicSystem] Choosing", ret_val)
        return ret_val

    def when_to_checkpoint(self, state: WorkloadState) -> int:
        weights = self._weights_for(state.request_number + 1)
        print(
            self.weights,
            weights,
            "Num Weights: ",
            len(weights),
            "Req Num: ",
            state.request_number,
            "Workload Dict: ",
            self.workload.__dict__,
        )
        interval = list(
            range(state.request_number + 1, state.request_number + len(weights) + 1)
        )
        if not interval:
            return 50000  # do not use a checkpoint
        weights = [self._weights_for(i, scalar=True) for i in interval]
        desired_request = random.choices(
            interval,
            weights=weights,
            k=1,
        )[0]

        print(f"[DynamicSystem] Checkpointing at {desired_request}",
              list(zip(interval, weights)))
        fixed_strat = FixedStrategy(self.workload, self.pool, desired_request)
        return fixed_strat.when_to_checkpoint(state)

    def on_request(self, state: WorkloadState):
        request_num = state.request_number - 1
        latest_latency = state.latencies[-1]

        # --- Exponentially-smoothed weight update (same as RequestCentric) ---
        cur_weight = self.weights[request_num]
        if cur_weight == 0:
            self.weights[request_num] = latest_latency
        else:
            self.weights[request_num] = (
                self.eps * latest_latency + (1 - self.eps) * cur_weight
            )
        self.weights[-1] = self.weights[-2]

        # --- Dynamic pool sizing ---
        self._recent_latencies.append(latest_latency)
        # Keep only the most recent `variance_window` observations
        if len(self._recent_latencies) > self.variance_window:
            self._recent_latencies = self._recent_latencies[-self.variance_window :]
        self._update_effective_capacity()

    def reset(self) -> None:
        for chkpt in self.pool:
            chkpt.delete()
        self.pool[:] = []
        self.weights = np.array([0] * self.workload.max_requests)
        self._recent_latencies = []
        self._effective_capacity = self.base_pool_size

    @property
    def extra_state(self) -> dict:
        return {
            "max_pool_size": self.max_pool_size,
            "min_pool_size": self.min_pool_size,
            "base_pool_size": self.base_pool_size,
            "p": self.p,
            "gamma": self.gamma,
            "weights": self.weights.tolist(),
            "eps": self.eps,
            "low_var_thresh": self.low_var_thresh,
            "high_var_thresh": self.high_var_thresh,
            "variance_window": self.variance_window,
            "recent_latencies": self._recent_latencies,
            "effective_capacity": self._effective_capacity,
        }

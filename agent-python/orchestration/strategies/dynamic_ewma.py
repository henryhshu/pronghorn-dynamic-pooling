from typing import List, Union

from .. import Parameters, CRStrategy, Checkpoint, WorkloadState, FixedStrategy
import random
import numpy as np
import copy

DEFAULT_MAX_POOL_SIZE = 20
DEFAULT_MIN_POOL_SIZE = 2
DEFAULT_BASE_POOL_SIZE = 8

DEFAULT_P = 0.40
DEFAULT_GAMMA = 0.10

DEFAULT_PERFORMANCE_FN = lambda arr: np.mean(np.array(arr))

DEFAULT_EPSILON = 0.5
MIN_WEIGHT_EPSILON = 1  # microseconds (median latency)

# Dual-rate EWMA decay parameters.
# alpha_fast controls how quickly the "recent volatility" tracker reacts.
# alpha_slow controls how quickly the "baseline volatility" tracker adapts.
# A large gap between the two lets the strategy distinguish transient spikes
# from inherent function variance.
DEFAULT_ALPHA_FAST = 0.30
DEFAULT_ALPHA_SLOW = 0.05

# Pool-sizing thresholds expressed as the ratio fast_dev / slow_dev.
#   ratio ≤ stable_threshold  →  min_pool_size  (unusually stable)
#   ratio ≥ spike_threshold   →  max_pool_size  (variance spike)
#   in between                →  linearly interpolated
DEFAULT_STABLE_THRESHOLD = 0.5
DEFAULT_SPIKE_THRESHOLD = 2.0


class DynamicEWMAStrategy(CRStrategy):
    """A CRStrategy variant with EWMA-driven dynamic pool sizing.

    Unlike ``DynamicSystemStrategy`` which uses the raw coefficient of
    variation (CV) of a sliding window, this strategy uses **dual-rate
    exponentially weighted moving averages (EWMAs)** to separate transient
    variance spikes from a function's inherent baseline volatility.

    Two EWMAs of the absolute latency deviation are maintained:

    * ``ewma_dev_fast`` — fast-decaying, tracks **recent** volatility.
    * ``ewma_dev_slow`` — slow-decaying, tracks **baseline** volatility.

    The pool-sizing signal is the *ratio* ``fast / slow``:

    * **ratio ≈ 1.0** → variance is normal for this function → moderate pool.
    * **ratio > spike_threshold** → variance is spiking above the
      function's baseline → grow the pool toward ``max_pool_size``.
    * **ratio < stable_threshold** → variance is unusually low → shrink.

    This design avoids penalising functions with inherently high but *stable*
    variance: because both EWMAs will be large, the ratio stays near 1.0.
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
        alpha_fast: float = DEFAULT_ALPHA_FAST,
        alpha_slow: float = DEFAULT_ALPHA_SLOW,
        stable_threshold: float = DEFAULT_STABLE_THRESHOLD,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
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

        # EWMA state
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.stable_threshold = stable_threshold
        self.spike_threshold = spike_threshold

        self._ewma_mean: float = 0.0      # EWMA of raw latencies
        self._ewma_dev_fast: float = 0.0   # Fast EWMA of |deviation|
        self._ewma_dev_slow: float = 0.0   # Slow EWMA of |deviation|
        self._n_observations: int = 0      # Number of latencies seen

        # Current effective capacity (starts at the base)
        self._effective_capacity: int = base_pool_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            f"DynamicEWMA_Max{self.max_pool_size}_Min{self.min_pool_size}"
            f"_P{self.p}_Gamma{self.gamma}"
        )

    @property
    def strategy(self) -> str:
        return "DynamicEWMA"

    @property
    def effective_capacity(self) -> int:
        """The current pool capacity derived from EWMA volatility ratio."""
        return self._effective_capacity

    # ------------------------------------------------------------------
    # EWMA helpers
    # ------------------------------------------------------------------

    def _update_ewma(self, latency: float) -> None:
        """Update all EWMA accumulators with a new latency observation."""
        self._n_observations += 1

        if self._n_observations == 1:
            # Bootstrap: first observation seeds all accumulators.
            self._ewma_mean = latency
            self._ewma_dev_fast = 0.0
            self._ewma_dev_slow = 0.0
            return

        # Update EWMA of the mean latency (using the fast alpha for
        # responsiveness; the mean is only used to derive deviation).
        self._ewma_mean = (
            self.alpha_fast * latency
            + (1 - self.alpha_fast) * self._ewma_mean
        )

        # Absolute deviation from the running mean
        deviation = abs(latency - self._ewma_mean)

        # Dual-rate EWMA of the deviation
        self._ewma_dev_fast = (
            self.alpha_fast * deviation
            + (1 - self.alpha_fast) * self._ewma_dev_fast
        )
        self._ewma_dev_slow = (
            self.alpha_slow * deviation
            + (1 - self.alpha_slow) * self._ewma_dev_slow
        )

    def _compute_volatility_ratio(self) -> float:
        """Return fast_dev / slow_dev.

        Returns 1.0 (neutral) when insufficient data or when the slow
        baseline is effectively zero.
        """
        if self._n_observations < 2:
            return 1.0
        if self._ewma_dev_slow < 1e-9:
            # Baseline deviation is essentially zero — treat as neutral.
            return 1.0
        return self._ewma_dev_fast / self._ewma_dev_slow

    def _update_effective_capacity(self) -> None:
        """Recalculate the effective pool capacity from the volatility ratio."""
        ratio = self._compute_volatility_ratio()

        if ratio <= self.stable_threshold:
            new_capacity = self.min_pool_size
        elif ratio >= self.spike_threshold:
            new_capacity = self.max_pool_size
        else:
            # Linear interpolation between min and max based on ratio
            t = (ratio - self.stable_threshold) / (
                self.spike_threshold - self.stable_threshold
            )
            new_capacity = round(
                self.min_pool_size
                + t * (self.max_pool_size - self.min_pool_size)
            )

        # Hard clamp to [min, max] (safety net)
        new_capacity = max(self.min_pool_size, min(new_capacity, self.max_pool_size))

        if new_capacity != self._effective_capacity:
            print(
                f"[DynamicEWMA] ratio={ratio:.4f} → adjusting capacity "
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
        by_performance = sorted(
            self.pool,
            key=lambda c: self._weights_for(c.state.request_number, scalar=True),
            reverse=True,
        )

        keeping_p = round(self.p * len(by_performance))
        output += by_performance[:keeping_p]
        by_performance = by_performance[keeping_p:]

        keeping_gamma = round(self.gamma * len(by_performance))
        output += random.choices(
            by_performance, k=min(keeping_gamma, len(by_performance))
        )

        output_chkpts = {chkpt for chkpt in output}
        removed = [chkpt for chkpt in self.pool if chkpt not in output_chkpts]
        for chkpt in removed:
            chkpt.delete()

        self.pool[:] = output
        print(
            f"[DynamicEWMA] Evicted to {len(self.pool)} checkpoints "
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

        print(f"[DynamicEWMA] Selecting checkpoint (pool={len(self.pool)}, "
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
        print("[DynamicEWMA] Choosing", ret_val)
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

        print(f"[DynamicEWMA] Checkpointing at {desired_request}",
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

        # --- EWMA-based dynamic pool sizing ---
        self._update_ewma(latest_latency)
        self._update_effective_capacity()

    def reset(self) -> None:
        for chkpt in self.pool:
            chkpt.delete()
        self.pool[:] = []
        self.weights = np.array([0] * self.workload.max_requests)
        self._ewma_mean = 0.0
        self._ewma_dev_fast = 0.0
        self._ewma_dev_slow = 0.0
        self._n_observations = 0
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
            "alpha_fast": self.alpha_fast,
            "alpha_slow": self.alpha_slow,
            "stable_threshold": self.stable_threshold,
            "spike_threshold": self.spike_threshold,
            "ewma_mean": self._ewma_mean,
            "ewma_dev_fast": self._ewma_dev_fast,
            "ewma_dev_slow": self._ewma_dev_slow,
            "n_observations": self._n_observations,
            "effective_capacity": self._effective_capacity,
        }

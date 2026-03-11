"""
PruningRequestCentricStrategy: request-centric pooling with latency/variance scoring
and two eviction modes (over-capacity replace vs converged shrink).

How the helpers fit together
----------------------------
1. ENTRY: orchestrator starts a container and needs a checkpoint (or cold start).
   It calls strategy.checkpoint_to_use() (orchestrator.py ~254).

2. checkpoint_to_use() (this file) does three things in order:
   a) EVICT IF OVER CAPACITY
      - _should_prune() -> True if pool > max_capacity, pool > min_pool_size, and
        latency is not "well enough" (max avg latency > baseline * latency_ratio_threshold).
      - If True: _prune_pool() runs. Sorts pool by _score_for_checkpoint (high score first),
        keeps top p% + gamma% random, never below min_pool_size, then deletes the rest.
   b) EVICT IF CONVERGED (optional shrink)
      - _should_shrink_converged() -> True if _is_variance_converged() and
        current pool size > _target_size_converged().
      - _is_variance_converged() uses EMA of mean(pool variances): converged when
        alpha <= converged_ratio * previous_ema; when alpha > previous_ema we don't shrink (grow again).
      - _target_size_converged() -> min_pool_size + round(ratio * buffer_max), where
        ratio = max_var/threshold and buffer_max is headroom or converged_shrink_buffer cap.
      - If True: _shrink_pool_converged() keeps top _target_size_converged() by score, deletes rest.
   c) CHOOSE WHICH CHECKPOINT TO USE (or None for cold start)
      - expanded_pool = pool + [None]. For each option, weight = _score_for_checkpoint(chkpt).
      - _score_for_checkpoint(chkpt) = base * (1/(avg+eps)) * (var+eps); base = _weights_for(req_num).
      - Softmax over weights, then random.choices(...) -> one checkpoint or None.
   Returns that checkpoint (or None).

3. _weights_for(req_num) (inherited from request_centric) uses learned self.weights
   (updated by on_request(latency)) to score request numbers; used as base in _score_for_checkpoint.

4. ELSEWHERE: on_request(state) updates self.weights from request latencies;
   when_to_checkpoint(state) picks the next request index to checkpoint at;
   register_checkpoint(chkpt) appends to pool when a new checkpoint is created.
"""
import random
from typing import List, Optional

import numpy as np

from .. import Checkpoint, Parameters
from .request_centric import (
    RequestCentricStrategy,
    DEFAULT_MAX_CAPACITY,
    DEFAULT_P,
    DEFAULT_GAMMA,
    DEFAULT_PERFORMANCE_FN,
    DEFAULT_EPSILON,
)

#Epsilons for combined score: avoid div-by-zero and keep scale sane
AVG_EPS = 1.0
VAR_EPS = 1.0

DEFAULT_MIN_POOL_SIZE = 3  #Don't prune below this; avoid over-shrinking when pool is small
#TODO: Hyperparams (need to tune)
DEFAULT_LATENCY_RATIO_THRESHOLD = 1.2  #Don't prune if max(avg_latency) <= baseline * this (everything "well enough")
# Adaptive variance convergence (EMA): alpha = mean(pool variances). EMA_new = ema_decay * prev_ema + ema_alpha * alpha.
# Converged when alpha <= converged_ratio * previous_ema (shrink). Grow again when alpha > previous_ema.
DEFAULT_VARIANCE_EMA_DECAY = 0.9
DEFAULT_VARIANCE_EMA_ALPHA = 0.1
DEFAULT_VARIANCE_CONVERGED_RATIO = 0.5  # alpha <= this * previous_ema -> converged
# Fallback scale for ratio when EMA not yet set (e.g. _target_size_converged)
DEFAULT_VARIANCE_CONVERGED_THRESHOLD = 100.0
# Max extra slots when variance is at threshold: None = use full headroom (max_capacity - min_pool_size)
DEFAULT_CONVERGED_SHRINK_BUFFER = None


class PruningRequestCentricStrategy(RequestCentricStrategy):
    """
    Subclass of RequestCentricStrategy with custom pruning:
    - Uses a single combined score (latency + variance) for ranking and selection.
    - When over capacity: replace/evict down to max_capacity (floor min_pool_size).
    - When all variances are low (converged): shrink toward a target size between
      min_pool_size and min_pool_size + buffer. Buffer size is variable: it scales
      with how close max variance is to the threshold (ratio in (0,1]) and with
      available headroom (max_capacity - min_pool_size), optionally capped by
      converged_shrink_buffer.
    """

    def __init__(
        self,
        workload: Parameters,
        pool: List[Checkpoint],
        max_capacity: int = DEFAULT_MAX_CAPACITY,
        min_pool_size: int = DEFAULT_MIN_POOL_SIZE,
        latency_ratio_threshold: float = DEFAULT_LATENCY_RATIO_THRESHOLD,
        variance_converged_threshold: float = DEFAULT_VARIANCE_CONVERGED_THRESHOLD,
        variance_ema_decay: float = DEFAULT_VARIANCE_EMA_DECAY,
        variance_ema_alpha: float = DEFAULT_VARIANCE_EMA_ALPHA,
        variance_converged_ratio: float = DEFAULT_VARIANCE_CONVERGED_RATIO,
        converged_shrink_buffer: Optional[int] = DEFAULT_CONVERGED_SHRINK_BUFFER,  # None = use full headroom
        p: float = DEFAULT_P,
        gamma: float = DEFAULT_GAMMA,
        performance_fn=DEFAULT_PERFORMANCE_FN,
        eps: float = DEFAULT_EPSILON,
    ) -> None:
        super().__init__(workload, pool, max_capacity=max_capacity, p=p, gamma=gamma, performance_fn=performance_fn, eps=eps)
        self.min_pool_size = min_pool_size
        self.latency_ratio_threshold = latency_ratio_threshold
        self.variance_converged_threshold = variance_converged_threshold
        self.variance_ema_decay = variance_ema_decay
        self.variance_ema_alpha = variance_ema_alpha
        self.variance_converged_ratio = variance_converged_ratio
        self.converged_shrink_buffer = converged_shrink_buffer
        self._variance_ema: Optional[float] = None  # EMA of mean(pool variances); updated when we check converged

    def _score_for_checkpoint(self, chkpt):
        """
        Combined score for a checkpoint: used for both pruning order and softmax selection.
        Higher score = prefer this checkpoint.

        Formula: score = base * (1 / (avg + eps)) * (var + eps)

        Comparison to base RequestCentricStrategy:
        - Base strategy uses only _weights_for(request_number): weight depends on the
          checkpoint's request index and global learned latencies (self.weights). Two
          checkpoints at the same request number get the same weight regardless of
          their own avg_response_time or var_response_time.
        - Here we keep that as 'base' but multiply by per-checkpoint (1/(avg+eps)) and
          (var+eps), so we prefer checkpoints that have been fast when measured and
          penalize "stably bad" (low-variance, slow) snapshots.

        Why base (self.weights) vs avg_response_time are not the same:
        - self.weights[req_num] is one number per request index: an exponential moving
          average of the latest latency seen at that index, updated by on_request()
          every time any container reports. So it's a global blend over time and runs.
        - avg_response_time is per checkpoint: mean(state.latencies) for that snapshot
          (the run that produced or used that checkpoint). So two checkpoints at the
          same request number can have different avg_response_times; base is the same
          for both. Using avg_response_time lets us rank those two snapshots by their
          own observed performance instead of only by the shared global weight.

        Why the extra factors:
        - base: request-centric quality (same as base strategy).
        - 1/(avg+eps): prefer checkpoints that have been fast when measured.
        - (var+eps): penalize low variance so we evict/favor less "stably bad" snapshots.

        Why multiply instead of add:
        - Multiplication means "good on all dimensions": if any factor is tiny (e.g.
          very high latency or very low variance), the whole score is tiny. So we
          don't rank a checkpoint high when it's bad on one axis. With addition, we'd
          need explicit weights and could still get high total from one strong term.
        - Each factor is a positive multiplier, so scale/units are less fragile than
          adding quantities that may live on different scales (base ~1e6/latency,
          latency_factor ~0.01, var ~100). Multiplication also matches "and" semantics:
          score = (good request index) and (fast snapshot) and (not stably bad).
        """
        if chkpt is None:
            return self._weights_for(0, scalar=True)
        base = self._weights_for(chkpt.state.request_number, scalar=True)
        latency_factor = 1.0 / (chkpt.avg_response_time + AVG_EPS)
        variance_factor = chkpt.var_response_time + VAR_EPS
        return base * latency_factor * variance_factor

    def _should_prune(self) -> bool:
        """True if we should run pruning (over capacity and not 'well enough')."""
        if len(self.pool) <= self.max_capacity:
            return False
        if len(self.pool) <= self.min_pool_size:
            return False
        avg_latencies = [c.avg_response_time for c in self.pool]
        baseline = float(np.min(avg_latencies))
        if baseline <= 0:
            return True  # avoid div by zero; allow prune
        if float(np.max(avg_latencies)) <= baseline * self.latency_ratio_threshold:
            return False
        return True

    def _is_variance_converged(self) -> bool:
        """
        Adaptive convergence: alpha = mean(pool variances). We keep EMA of alpha.
        Converged when alpha <= converged_ratio * previous_ema (variance dropped a lot).
        When alpha > previous_ema we are not converged (grow again). EMA updated every check.
        """
        if len(self.pool) == 0:
            return False
        alpha = float(np.mean([c.var_response_time for c in self.pool]))
        if self._variance_ema is None:
            self._variance_ema = alpha
            return False
        converged = alpha <= (self.variance_converged_ratio * self._variance_ema)
        self._variance_ema = self.variance_ema_decay * self._variance_ema + self.variance_ema_alpha * alpha
        return converged

    def _should_shrink_converged(self) -> bool:
        """True if variance has converged and weighted target size is below current size."""
        if not self._is_variance_converged():
            return False
        return len(self.pool) > self._target_size_converged()

    def checkpoint_to_use(self) -> Checkpoint:
        if self._should_prune():
            self._prune_pool()
        elif self._should_shrink_converged():
            self._shrink_pool_converged()
        # Use combined score (latency + variance) for softmax selection
        print("Exploiting")
        expanded_pool = self.pool + [None]
        weights = [self._score_for_checkpoint(chkpt) for chkpt in expanded_pool]
        weights = np.array(weights, dtype=float)
        weights_max = np.amax(weights, keepdims=True)
        weights_shifted = np.exp(weights - weights_max)
        weights = weights_shifted / np.sum(weights_shifted, keepdims=True)
        weights = weights.tolist()
        print(list(zip(weights, expanded_pool)))
        ret_val = random.choices(expanded_pool, weights=weights, k=1)[0]
        print("Choosing", ret_val)
        return ret_val

    def _prune_pool(self):
        if len(self.pool) == 0:
            return
        # Single equation: sort by combined score (penalizes high avg, low var)
        by_performance = sorted(
            self.pool,
            key=self._score_for_checkpoint,
            reverse=True,
        )
        keeping_p = round(self.p * len(by_performance))
        output = list(by_performance[:keeping_p])
        by_performance = by_performance[keeping_p:]
        keeping_gamma = round(self.gamma * len(by_performance))
        output += random.choices(
            by_performance, k=min(keeping_gamma, len(by_performance))
        )
        # Never shrink below min_pool_size
        if len(output) < self.min_pool_size and len(by_performance) > 0:
            need = self.min_pool_size - len(output)
            extra = [c for c in by_performance if c not in output]
            output += extra[:need]
        output_chkpts = set(output)
        removed = [c for c in self.pool if c not in output_chkpts]
        for chkpt in removed:
            chkpt.delete()
        self.pool[:] = output
        print(
            f"Evicted all but top {keeping_p} by score (latency+var) and {keeping_gamma} by random"
        )
        assert len(self.pool) <= self.max_capacity

    def _target_size_converged(self) -> int:
        """
        Target pool size when variance has converged: between min_pool_size and
        min_pool_size + buffer. Ratio uses EMA when set (max_var / _variance_ema), else
        fallback to variance_converged_threshold.
        """
        max_var = float(np.max([c.var_response_time for c in self.pool]))
        scale = self._variance_ema if (self._variance_ema is not None and self._variance_ema > 0) else self.variance_converged_threshold
        if scale <= 0:
            return self.min_pool_size
        ratio = min(1.0, max_var / scale)
        headroom = max(0, self.max_capacity - self.min_pool_size)
        buffer_max = headroom if self.converged_shrink_buffer is None else min(self.converged_shrink_buffer, headroom)
        extra = round(ratio * buffer_max)
        target = self.min_pool_size + extra
        return max(self.min_pool_size, min(target, len(self.pool), self.max_capacity))

    def _shrink_pool_converged(self):
        """
        All variances are low (converged): shrink pool toward a weighted target size
        (between min_pool_size and min_pool_size + buffer), keeping top by score.
        """
        if len(self.pool) <= self.min_pool_size:
            return
        target_size = self._target_size_converged()
        if target_size >= len(self.pool):
            return
        by_performance = sorted(
            self.pool,
            key=self._score_for_checkpoint,
            reverse=True,
        )
        output = list(by_performance[:target_size])
        output_chkpts = set(output)
        removed = [c for c in self.pool if c not in output_chkpts]
        for chkpt in removed:
            chkpt.delete()
        self.pool[:] = output
        print(
            f"Variance converged (EMA): shrunk pool from {len(by_performance)} to {target_size} "
            f"(target=min+{target_size - self.min_pool_size})"
        )

    @property
    def extra_state(self) -> dict:
        return {
            **super().extra_state,
            "min_pool_size": self.min_pool_size,
            "latency_ratio_threshold": self.latency_ratio_threshold,
            "variance_converged_threshold": self.variance_converged_threshold,
            "variance_ema_decay": self.variance_ema_decay,
            "variance_ema_alpha": self.variance_ema_alpha,
            "variance_converged_ratio": self.variance_converged_ratio,
            "converged_shrink_buffer": self.converged_shrink_buffer,
            "variance_ema": self._variance_ema,
        }

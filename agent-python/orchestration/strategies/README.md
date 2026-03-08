# Pronghorn Orchestration Strategies

This directory contains the checkpoint/restore (CR) orchestration strategies that Pronghorn uses to decide **when** to create snapshots, **which** snapshot to serve, and **how many** snapshots to keep in the pool.

All strategies extend the [`CRStrategy`](../strategy.py) base class.

---

## Available Strategies

| Strategy | File | Pool Sizing | Description |
|---|---|---|---|
| **ColdStart** | `cold_start.py` | N/A (no pool) | Always starts from scratch. Useful as a baseline. |
| **Fixed** | `fixed.py` | Fixed | Snapshots at a predetermined request number. Good for deterministic workloads. |
| **RequestCentric** | `request_centric.py` | Fixed (`max_capacity`) | Learns per-request latency weights and uses them to intelligently select and time snapshots. Pool is bounded by a static `max_capacity`. |
| **DynamicSystem** | `dynamic_system.py` | **Dynamic** (CV-driven) | Extends RequestCentric's weight-based logic with a pool size that dynamically converges to a local (system-level) optimum based on the CV of recent request latencies. |
| **DynamicEWMA** *(new)* | `dynamic_ewma.py` | **Dynamic** (EWMA-driven) | Uses dual-rate EWMAs to compare recent volatility against baseline volatility. Avoids penalising functions with inherently high but stable variance. |

---

## DynamicSystem Strategy

### Motivation

Snapshots are **memory-intensive**. The existing `RequestCentricStrategy` maintains a fixed pool cap (`max_capacity`), which means:

- If the JIT compiler has already optimized the function (low latency variance), we waste memory keeping a large pool.
- If incoming requests have highly variable latencies, a small pool may not provide enough snapshot diversity.

`DynamicSystemStrategy` solves this by **dynamically scaling the pool capacity** between a configurable minimum and maximum based on the **coefficient of variation (CV)** of recent request latencies:

```
CV = standard_deviation(latencies) / mean(latencies)
```

> **Note:** This strategy converges toward a *local (system-level) optimum* rather than a global one — the pool size adapts to the observed latency variance of the current workload, not a globally optimal configuration.

### How It Works

```
                   Low CV                    High CV
                  (stable)                 (volatile)
    ┌──────────────┼───────────────────────────┼──────────────┐
    │ min_pool_size│     linear interpolation   │max_pool_size │
    └──────────────┼───────────────────────────┼──────────────┘
              low_var_thresh           high_var_thresh
                  (0.10)                    (1.0)
```

1. **On every request**, `on_request()` records the latest latency and recomputes the CV over a sliding window of recent observations.
2. The **effective pool capacity** is then updated:
   - `CV ≤ low_var_thresh` → capacity = `min_pool_size`
   - `CV ≥ high_var_thresh` → capacity = `max_pool_size`
   - In between → capacity is **linearly interpolated**
3. When `checkpoint_to_use()` is called, if the pool exceeds the effective capacity, `_prune_pool()` evicts checkpoints using the same p/gamma eviction policy as `RequestCentricStrategy`.
4. The `max_pool_size` is an **absolute hard cap** that prevents memory from exploding regardless of variance.

### Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_pool_size` | `20` | Absolute maximum number of snapshots allowed in the pool. |
| `min_pool_size` | `2` | Minimum pool size even when variance is very low. |
| `base_pool_size` | `8` | Initial pool capacity before enough latency data is collected. |
| `p` | `0.40` | Fraction of top-performing checkpoints to keep during pruning. |
| `gamma` | `0.10` | Fraction of remaining checkpoints to keep randomly during pruning. |
| `eps` | `0.5` | Exponential smoothing factor for per-request weight updates. |
| `low_var_thresh` | `0.10` | CV at or below this value maps to `min_pool_size`. |
| `high_var_thresh` | `1.0` | CV at or above this value maps to `max_pool_size`. |
| `variance_window` | `50` | Number of recent latencies to consider when computing CV. |

### Usage

#### Constructing Directly

```python
from orchestration import DynamicSystemStrategy, Parameters

strategy = DynamicSystemStrategy(
    workload=Parameters(),
    pool=[],
    max_pool_size=20,
    min_pool_size=2,
    base_pool_size=8,
    low_var_thresh=0.10,
    high_var_thresh=1.0,
    variance_window=50,
)
```

#### Using via ENV variable

Set the first component of the `ENV` environment variable to `dynamic_system`:

```bash
export ENV="dynamic_system,500,10"
#            ^strategy     ^eviction ^max_requests (parsed elsewhere)
```

The `cr_deserialize()` function in [`utils.py`](../utils.py) will automatically instantiate a `DynamicSystemStrategy` with default parameters when it sees this value.

#### Serialization / Deserialization

`DynamicSystemStrategy` implements `extra_state` so its full state (weights, recent latencies, and effective capacity) is serialized alongside the common strategy state. The `cr_deserialize()` function handles restoring all state when loading from a serialized JSON payload.

### Example Scenarios

**Scenario 1: Stable workload (low variance)**
```
Latencies: [100, 102, 99, 101, 100, ...]   →   CV ≈ 0.01
Pool capacity shrinks to min_pool_size (2)
Memory savings: significant
```

**Scenario 2: Bursty workload (high variance)**
```
Latencies: [50, 500, 80, 1200, 60, ...]    →   CV ≈ 1.5
Pool capacity grows to max_pool_size (20)
More snapshot diversity to handle unpredictable latency patterns
```

**Scenario 3: Moderate variance**
```
Latencies: [100, 150, 120, 180, 110, ...]  →   CV ≈ 0.25
Pool capacity = 2 + ((0.25 - 0.10) / (1.0 - 0.10)) × (20 - 2) ≈ 5
Balanced tradeoff between memory and snapshot coverage
```

---

## DynamicEWMA Strategy

### Motivation

`DynamicSystemStrategy` uses the raw coefficient of variation (CV) of a sliding window of latencies. This works well when variance spikes are transient, but **penalises functions with inherently high variance** — they always get a large pool even when their variance is normal for them.

`DynamicEWMAStrategy` fixes this by using **dual-rate exponentially weighted moving averages (EWMAs)** to separate *transient spikes* from *inherent baseline volatility*.

### How It Works

Two EWMAs of the absolute latency deviation from the running mean are maintained:

- **`ewma_dev_fast`** (decay `alpha_fast=0.30`) — tracks *recent* volatility, reacts quickly to changes.
- **`ewma_dev_slow`** (decay `alpha_slow=0.05`) — tracks *baseline* volatility, adapts slowly.

The pool-sizing signal is the **ratio**:

```
volatility_ratio = ewma_dev_fast / ewma_dev_slow
```

```
                ratio ≤ 0.5               ratio ≥ 2.0
               (stable)                  (spike)
  ┌──────────────┼────────────────────────┼──────────────┐
  │ min_pool_size│   linear interpolation  │max_pool_size │
  └──────────────┼────────────────────────┼──────────────┘
           stable_threshold          spike_threshold
```

**Key insight**: A function with consistently high variance has *both* EWMAs at high values, so the ratio stays near **1.0** → moderate pool. Only when variance *spikes above the function's own baseline* does the ratio exceed the spike threshold.

### Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_pool_size` | `20` | Absolute maximum pool size. |
| `min_pool_size` | `2` | Minimum pool size. |
| `base_pool_size` | `8` | Initial pool capacity before enough data is collected. |
| `alpha_fast` | `0.30` | Decay rate for the fast (recent) EWMA. |
| `alpha_slow` | `0.05` | Decay rate for the slow (baseline) EWMA. |
| `stable_threshold` | `0.5` | Ratio at or below which the pool shrinks to min. |
| `spike_threshold` | `2.0` | Ratio at or above which the pool grows to max. |
| `p` | `0.40` | Fraction of top-performing checkpoints to keep during pruning. |
| `gamma` | `0.10` | Fraction of remaining checkpoints to keep randomly. |
| `eps` | `0.5` | Exponential smoothing factor for per-request weight updates. |

### Usage

#### Constructing Directly

```python
from orchestration import DynamicEWMAStrategy, Parameters

strategy = DynamicEWMAStrategy(
    workload=Parameters(),
    pool=[],
    max_pool_size=20,
    min_pool_size=2,
    base_pool_size=8,
    alpha_fast=0.30,
    alpha_slow=0.05,
    stable_threshold=0.5,
    spike_threshold=2.0,
)
```

#### Using via ENV variable

```bash
export ENV="dynamic_ewma,500,10"
```

### Memory Advantage

Unlike `DynamicSystemStrategy` which stores a window of recent latencies, `DynamicEWMAStrategy` only stores **5 scalar values** (`ewma_mean`, `ewma_dev_fast`, `ewma_dev_slow`, `n_observations`, `effective_capacity`), making it more memory-efficient for serialization across workers.

---

## DynamicSystem vs DynamicEWMA

| Aspect | DynamicSystem | DynamicEWMA |
|---|---|---|
| **Signal** | CV of sliding window | EWMA fast/slow ratio |
| **Memory** | O(window_size) latencies | O(1) — 5 scalars |
| **Inherently high-variance functions** | Gets large pool (penalised) | Gets moderate pool (ratio ≈ 1.0) |
| **Reaction to spikes** | Depends on window fill | Immediate via fast EWMA |
| **Smoothness** | Step changes as data exits window | Smooth exponential decay |

---

## CRStrategy Interface

All strategies must implement the following methods from the [`CRStrategy`](../strategy.py) base class:

| Method | Purpose |
|---|---|
| `checkpoint_to_use() → Checkpoint` | Select which snapshot to restore from for the next invocation. |
| `when_to_checkpoint(state) → int` | Determine the request number at which to create a new snapshot. |
| `on_request(state)` | Called after each request — use to update internal state (weights, latencies, etc.). |
| `on_eviction(checkpoint, final_state)` | Called when a container is evicted. |
| `reset()` | Clear all strategy state and delete pooled checkpoints. |
| `extra_state → dict` | Return strategy-specific state for serialization. |

---

## Testing

Run all strategy tests:

```bash
cd agent-python
python -m pytest tests/ -v
```

Run individually:

```bash
python -m pytest tests/test_dynamic_system.py -v   # 34 tests
python -m pytest tests/test_dynamic_ewma.py -v     # 36 tests
```

# Cost Analysis

This directory contains analysis notebooks and scripts for evaluating Pronghorn's orchestration strategies.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| [`evaluation_analysis.ipynb`](evaluation_analysis.ipynb) | **Original** full-suite analysis across all 13 benchmarks (9 PyPy + 4 JVM) and 3 strategies. Compares `request_centric` against `fixed` baseline. |
| [`dynamic_system_analysis.ipynb`](dynamic_system_analysis.ipynb) | **New** focused analysis comparing the `dynamic_system` variance-driven pool strategy against all baselines on BFS and MST. |

---

## Dynamic System Analysis Notebook

### Overview

The `dynamic_system_analysis.ipynb` notebook is designed for comparing the **DynamicSystem** strategy against Pronghorn's baseline orchestration strategies. It focuses on two representative PyPy serverless functions (**BFS** and **MST**) and produces:

1. **Convergence analysis** — request number at which each strategy reaches steady-state latency
2. **Median latency comparison** — improvement percentages vs the baseline
3. **Rate analysis** — performance across eviction rates (1, 4, 20)
4. **CDF visualisations** — cumulative distribution plots of client-side latency

### Modular Strategy Configuration

Strategies are defined in a single `STRATEGIES` dictionary at the top of the notebook:

```python
STRATEGIES = {
    'Cold Start':       'cold',
    'Fixed':            'fixed&request_to_checkpoint=1',
    'Request Centric':  'request_centric&max_capacity=12',
    'Dynamic System':   'dynamic_system',
}
```

**To add a new strategy:**

1. Add a new entry to `STRATEGIES` with a human-readable label as key and the ENV/CSV identifier as value:
   ```python
   'My New Strategy':  'my_new_strategy&param=value',
   ```
2. Make sure the CSV data for that strategy exists in `../data/` (produced by the corresponding `synthetic_run*.py` script).
3. Re-run the notebook — all analysis cells automatically include the new strategy.

**To change the baseline or evaluation target:**

```python
BASELINE_STRATEGY_LABEL = 'Fixed'         # strategy to compare against
EVAL_STRATEGY_LABEL = 'Dynamic System'    # strategy being evaluated
```

### Adding Benchmarks

Edit the `BENCHMARKS` list:

```python
BENCHMARKS = ['bfs', 'mst']  # Add more here, e.g. 'dfs', 'pagerank'
```

If adding a new benchmark, also add a display name in `BENCHMARK_TITLES`.

### Enabling Java/JVM Analysis

Set the flag at the top of the configuration cell:

```python
INCLUDE_JAVA = True
```

This will load `java-evaluation.csv` and `java-evaluation-dynamic-system.csv` and append the JVM benchmarks to the analysis.

### Data Requirements

The notebook expects CSV files in `../data/` with columns:

```
request_number, benchmark, mutability, strategy, rate, client, server, overhead
```

| File | Produced by |
|---|---|
| `python-evaluation.csv` | `run.sh evaluation` (original baselines) |
| `python-evaluation-dynamic-system.csv` | `run_dynamic_system.sh evaluation` (includes `dynamic_system`) |
| `java-evaluation.csv` *(optional)* | `run.sh evaluation` |
| `java-evaluation-dynamic-system.csv` *(optional)* | `run_dynamic_system.sh evaluation` |

### Running the Notebook

```bash
cd cost-analysis
jupyter notebook dynamic_system_analysis.ipynb
```

Or via command line:

```bash
jupyter nbconvert --to notebook --execute dynamic_system_analysis.ipynb
```

### Output

- **Console tables**: Convergence points, median latencies, improvement categorisation
- **`cdf_dynamic_system.png`**: CDF plot comparing all strategies for each benchmark

---

## Other Files

| File | Description |
|---|---|
| `table_4.py` | Computes checkpoint/restore overhead metrics (Table 4 in the paper). |
| `table_4_results.json` | Pre-computed results from a prior evaluation run. |
| `table_5.py` | Computes storage and network bandwidth usage (Table 5 in the paper). |
| `table_5_results.txt` | Pre-computed results from a prior evaluation run. |

---

## Key Changes from Original Notebook

| Aspect | `evaluation_analysis.ipynb` | `dynamic_system_analysis.ipynb` |
|---|---|---|
| **Benchmarks** | All 13 (9 PyPy + 4 JVM) | BFS + MST (PyPy), Java optional |
| **Strategies** | 3 (cold, fixed, request_centric) | 4 (+ dynamic_system), modular registry |
| **Baseline** | `fixed` (hardcoded) | Configurable via `BASELINE_STRATEGY_LABEL` |
| **Evaluation target** | `request_centric` (hardcoded) | Configurable via `EVAL_STRATEGY_LABEL` |
| **Adding strategies** | Requires editing multiple cells | Add one dict entry in config cell |
| **Java/JVM** | Always included | Optional (`INCLUDE_JAVA` flag) |
| **Visualisations** | None in notebook | CDF plots with auto-save |

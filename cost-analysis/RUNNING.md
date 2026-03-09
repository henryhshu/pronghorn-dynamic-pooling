# Cost analysis – step-by-step running instructions

This guide explains how to run the cost-analysis notebooks and scripts, and **what to change when you have only run a subset of experiments** (e.g. only BFS, only rate 20, or only certain strategies).

---

## Prerequisites

1. **Data directory**  
   The notebooks expect CSV files in `../data/` (relative to the `cost-analysis` folder). Create it if needed:
   ```bash
   mkdir -p ../data
   ```
2. **Python**  
   Use a environment with: `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, `jupyter`.

   From the repo root:
   ```bash
   pip install pandas numpy scipy seaborn matplotlib jupyter
   ```

---

## 1. Full run (all benchmarks, all rates, all strategies)

If you have run the full evaluation and have:

- `../data/python-evaluation.csv`
- `../data/java-evaluation.csv` (optional for `evaluation_analysis`)

then:

1. Open **`evaluation_analysis.ipynb`**.
2. Run all cells in order (Kernel → Run All).  
   The first “Load data” cell will print which benchmarks, rates, and strategies are present.

For **`dynamic_system_analysis.ipynb`** you also need:

- `../data/python-evaluation.csv` (baseline strategies)
- `../data/python-evaluation-dynamic-system.csv` (includes `dynamic_system`)

1. Open **`dynamic_system_analysis.ipynb`**.
2. Run all cells in order.

---

## 2. Only BFS (no MST or other benchmarks)

### evaluation_analysis.ipynb

1. Open the **Experiment constants** cell (the one with `platforms`, `eviction_rates`, etc.).
2. Set the PyPy benchmark list to only BFS and, if you did not run Java, clear JVM:
   ```python
   platforms = {
       "pypy": ['bfs'],
       "jvm": []   # or omit Java benchmarks you didn't run
   }
   ```
3. Run the **Load data** cell, then the rest of the notebook.  
   Convergence and later sections will only use BFS (and any Java benchmarks you left in).

### dynamic_system_analysis.ipynb

1. Open the **Configuration** cell (Section 1).
2. Set:
   ```python
   BENCHMARKS = ['bfs']
   ```
3. Run all cells. CDF and tables will show only BFS.

---

## 3. Only rate 20 (no rate 1 or 4)

### evaluation_analysis.ipynb

1. In the **Experiment constants** cell, set:
   ```python
   eviction_rates = [20]
   ```
2. Run from the top.  
   - The convergence table (rate 4) will only show a column for rate 4; if you have no rate-4 data, that column may be empty or show `None`.  
   - The “Request Rates” section will only have one rate (20).

### dynamic_system_analysis.ipynb

1. In the **Configuration** cell, set:
   ```python
   EVICTION_RATES = [20]
   ```
2. Run all cells. Rate-based tables and plots will only use rate 20.

---

## 4. Only certain strategies

### evaluation_analysis.ipynb

1. In the **Experiment constants** cell, set `strategies` to the exact strategy names as they appear in your CSV (e.g. in the `strategy` column):
   ```python
   strategies = ['cold', 'request_centric&max_capacity=12']
   ```
   Or, if you only ran one strategy and cold:
   ```python
   strategies = ['cold', 'fixed&request_to_checkpoint=1']
   ```
2. **Important:** The “Orchestration Strategy” and “Request Rates” sections compute **improvement vs fixed** and **vs request_centric**. So:
   - For the printed improvement and geometric-mean cells to work, your data should contain **both** `fixed&request_to_checkpoint=1` and `request_centric&max_capacity=12`.
   - If you only ran, say, `cold` and `request_centric&max_capacity=12`, those improvement cells will warn and may show NaNs or fail; in that case either run the fixed strategy as well or skip/comment out the cells that assume both.

### dynamic_system_analysis.ipynb

1. In the **Configuration** cell, edit **`STRATEGIES`** so it only contains the strategies you actually ran and have in the CSV:
   ```python
   STRATEGIES = {
       'Cold Start':       'cold',
       'Fixed':            'fixed&request_to_checkpoint=1',
       'Request Centric':  'request_centric&max_capacity=12',
       # 'Dynamic System':   'dynamic_system',  # comment out if not in data
   }
   ```
2. Set **`BASELINE_STRATEGY_LABEL`** and **`EVAL_STRATEGY_LABEL`** to two strategy labels that exist in `STRATEGIES` and in your data.
3. Run all cells. Any strategy missing from the CSV will simply have no points in the CDF and no row in the comparison tables.

---

## 5. Combined subset (e.g. only BFS + rate 20 + two strategies)

Apply the relevant edits from sections 2–4 together:

**evaluation_analysis.ipynb:**

- `platforms = {"pypy": ['bfs'], "jvm": []}`
- `eviction_rates = [20]`
- `strategies = ['cold', 'fixed&request_to_checkpoint=1', 'request_centric&max_capacity=12']` (or whatever subset you ran)

**dynamic_system_analysis.ipynb:**

- `BENCHMARKS = ['bfs']`
- `EVICTION_RATES = [20]`
- `STRATEGIES` with only the strategies you have in the CSVs.

Then run the notebooks from the top.

---

## 6. Table 4 and Table 5 scripts

- **`table_4.py`**  
  Measures CRIU dump/restore times and checkpoint size (Table 4). It is **Linux-specific** (uses `criu`, `pgrep`, etc.). Not required for the notebooks.

- **`table_5.py`**  
  Reads **`table_4_results.json`** and computes storage and network overhead (Table 5).  
  - Run from the `cost-analysis` directory:  
    `python table_5.py`  
  - If you only have BFS (or only MST), keep only the corresponding entries in `table_4_results.json`; the script will still run and report only those benchmarks.

---

## 7. Quick checklist

| What you ran              | What to change |
|---------------------------|----------------|
| Only BFS                  | `platforms["pypy"] = ['bfs']` and/or `BENCHMARKS = ['bfs']` |
| Only MST                  | `platforms["pypy"] = ['mst']` and/or `BENCHMARKS = ['mst']` |
| Only rate 20              | `eviction_rates = [20]` and/or `EVICTION_RATES = [20]` |
| Only some strategies      | Set `strategies` / `STRATEGIES` to match your CSV; ensure baseline and eval strategy exist if you use improvement cells |
| No Java                   | `platforms["jvm"] = []` and/or `INCLUDE_JAVA = False` |
| No dynamic_system data    | In `dynamic_system_analysis`, remove `'Dynamic System'` from `STRATEGIES` and set `EVAL_STRATEGY_LABEL` to something you have |

Running the **Load data** (or data-loading) cell first will show which benchmarks, rates, and strategies are present in your CSVs; use that to align the config with your actual data.

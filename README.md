AUD-USD Foreign Exchange Market Simulation Based on Agents

This project implements a multi-agent (ABM) simulation of the AUD/USD FX market, including speculators, hedgers, fundamentalists, and a central bank intervention mechanism. The model uses an order-book matching process to generate price paths and compares them with real market data to observe stylized facts such as fat-tailed return distributions and volatility clustering.

You can run and visualize the simulation via a command-line script or a Jupyter Notebook. The project includes reproducibility utilities to keep results consistent across runs and environments.

Highlights
- Multi-agent market: Speculators, Hedgers, Fundamentalists, Central Bank
- Order-book matching: limit/market orders, trade price and clearing logic
- Dynamic factors: fundamental updates, external shocks, sentiment and momentum
- Reproducibility: unified random seed, environment and data fingerprint logging
- Visualization: price, returns, rolling volatility, Q-Q plot, real-vs-simulated comparison

Project Structure (key files)
- `src/forex_abm.py`: ABM core implementation and matching logic
- `test_model_improved.py`: CLI-friendly simulation and figure generation script
- `Demo.ipynb`: Notebook demo with one-click run and visualization
- `utils/reproducibility.py`: seed setup, environment info, data SHA256, run metadata logging
- `utils/visualization.py`: visualization helpers
- `data/merged_data.csv`: merged sample dataset (must include columns: `AUD_USD_Close`, `interest_diff`, `IronOre_Price`)
- `results/`: output directory (figures and statistics)

Environment
- Python 3.10+ (Windows/macOS/Linux)
- Install dependencies: `pip install -r requirements.txt`

Quick Start (script)
1) Set seed and run the simulation (PowerShell example)
   - `mkdir results\figures` (auto-created if missing)
   - `$env:SEED=42; python test_model_improved.py`
2) Outputs
   - `results/improved_simulation.csv`: price path
   - `results/simulation_statistics.csv`: key metrics
   - `results/figures/improved_simulation.png`: price/returns/rolling volatility
   - `results/figures/return_analysis.png`: return distribution and Q-Q plot
   - `results/figures/real_vs_simulated.png`: real-data comparison (if data available)
   - `results/run_metadata.json`: run metadata (see “Reproducibility”)

Notebook Demo
- Open `Demo.ipynb` and run the first three cells in order:
  - Set the random seed (override with env var `SEED`, default 42)
  - One-click run: invokes `test_model_improved.py` to generate results and `run_metadata.json`
  - Subsequent cells read files under `results/` for visualization and stats
- Alternatively, set the seed before starting the Notebook:
  - PowerShell: `$env:SEED=123; jupyter notebook`
  - Bash: `SEED=123 jupyter notebook`

Reproducibility
- Unified random seed
  - The seed is set at the top of `test_model_improved.py` (overridable via env var `SEED`).
  - `utils/reproducibility.set_seed(seed)` seeds both Python `random` and NumPy.
- Run metadata logging
  - Each run writes `results/run_metadata.json` with:
    - `seed`, `sim_steps`, start/end price, total return, total trade count
    - Python/OS and core library versions (NumPy, pandas, SciPy, Matplotlib)
    - SHA256 of input data file `data/merged_data.csv` (if present)
- Dependency locking (optional, recommended)
  - `pip freeze > requirements.lock.txt`
  - Recreate env: `pip install -r requirements.lock.txt`
- Threads / numeric libs (optional)
  - For stronger determinism, limit BLAS threads:
    - Bash: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
    - PowerShell: `$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'`

Data Notes
- By default, scripts read real data from `data/merged_data.csv`. If missing or incompatible, the script falls back to a default initial price.
- Required columns: `AUD_USD_Close`, `interest_diff`, `IronOre_Price`.
- The repo includes sample data files (`data/` and `src/data/`) for quick trials. For formal experiments, fix a data version or track SHA256.

FAQ
- ImportError/missing packages: ensure `pip install -r requirements.txt` has been run.
- Notebook install prompts: if auto-install runs, restart the kernel afterward.
- Missing data columns: ensure `data/merged_data.csv` includes required columns, or update column mappings in the script.
- Setting env vars in PowerShell: `$env:SEED=123` (valid for current session).

Project Production
- Course: CITS4403
- Team: Kunyu He (24213379), Lintao Gong (23943051)


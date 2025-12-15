
# EV Stag Hunt Model — Assignment 3

This repository contains an agent-based model of electric vehicle (EV) adoption framed as a **Stag Hunt coordination game**, implemented using the Mesa framework. The code is based on teaching materials by **Michael Lees**, with additional experiments, network comparisons, and policy interventions developed for Assignment 3.

---

## Installation

### 1. Python version

This project was developed and tested using:

```
Python 3.13.9
```

Check your Python version with:

```bash
python --version
```

---

### 2. Install dependencies

All required Python packages are listed in `requirements.txt`.

From the project root directory, run:

```bash
pip install -r requirements.txt
```

> If you are using a virtual environment, activate it before running the command above.

---

## Repository Structure

```
Assignment-3-MBDM/
├── ev_core.py            # Core Mesa model and agent logic
├── ev_experiments.py     # Experiment runners, sweeps, and policy logic
├── ev_plotting.py        # Plotting utilities (spaghetti, density, phase plots)
├── assignment.ipynb     # Main notebook running all analyses
├── plots/               # Output directory for generated figures
├── requirements.txt     # Python dependencies
└── README.md
```

Only the files listed above are required to run the experiments.

---

## How to Run the Experiments

All experiments for the assignment are executed from the Jupyter notebook:

```
assignment.ipynb
```

### Steps to reproduce results

1. Navigate to the project directory:

   ```bash
   cd Assignment-3-MBDM
   ```

2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `assignment.ipynb`

4. Run the notebook

The notebook is structured to run:

* Baseline system analysis (Part 1)
* Network structure comparison (Part 2)
* Policy intervention experiments (Part 3)

No additional scripts need to be run manually.

---

## Outputs

* All plots are automatically saved to the `plots/` directory.
* Output figures include:

  * Phase plots (parameter sweeps)
  * Ratio sweeps
  * Spaghetti plots
  * Density plots
  * Fan charts (baseline vs policy)
* Summary tables (CSV) are saved for policy evaluation metrics.

No manual post-processing is required.

---

## Notes on Reproducibility

* Random seeds are fixed where applicable.
* Due to stochastic dynamics, individual runs may vary slightly, but qualitative results are robust.
* All figures shown in the report can be reproduced by running `assignment.ipynb`.

---

## Acknowledgments

This project builds on teaching code provided by **Michael Lees**.
All extensions, experiments, analyses, and policy designs were implemented as part of **Assignment 3**.

---


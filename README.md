# Merging Universes Simulation

Physically rigorous model of merging universes as an alternative to ΛCDM. Dark energy is generated at the interface between overlapping universes.

## Setup

```bash
cd simulation
pip install -r requirements.txt
```

## Run

```bash
python run.py              # standard run with ΛCDM comparison
python run.py --quick      # fast test run
python run.py --mcmc       # MCMC parameter estimation
python run.py --no-plots   # skip figure generation
```

Output: console metrics (χ², BIC, AIC), figures in `simulation/figs/`, and JSON/npz results in `simulation/results/`.
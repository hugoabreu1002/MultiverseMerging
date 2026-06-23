# Merging Universes Simulation

Physically rigorous model of merging universes as an alternative to ΛCDM. Dark energy is generated at the interface between overlapping universes, and dark matter emerges from actual foreign matter crossing between merging universes.

## Two Competing Geometries

The model explores two ways the multiverse could be structured:

### Model A: 5D Spatial Brane
- Extra dimension is **spatial** (a 5th axis in the bulk)
- Universes are 3D branes separated along this axis
- They expand into each other → spatial overlap
- DE = overlap volume energy (sharp onset at contact)
- DM = foreign matter crossing the spatial gap

### Model B: 4D Temporal Quantum Multiverse
- No extra spatial dimension. The multiverse exists across **quantum time**
- Each universe is a quantum history/slice in the Wheeler-DeWitt wavefunction
- "Merging" = quantum interference between different time slices
- DE = constructive interference of vacuum energy across time (softer onset)
- DM = actual matter from adjacent time-slice universes leaking via quantum tunneling
- DM enhanced in regions with greater gravitational time dilation

## Setup

```bash
cd simulation
pip install -r requirements.txt
```

## Run

```bash
python run.py                    # Default 5D model (DE only)
python run.py --geometry 4D      # 4D temporal quantum model
python run.py --with-dm          # 5D + foreign dark matter
python run.py --geometry 4D --with-dm  # 4D + foreign DM

# Geometry comparison (runs all 4 variants)
python run_comparison.py

# Standard run with ΛCDM comparison
python run.py --quick
python run.py --mcmc             # MCMC parameter estimation
python run.py --no-plots         # skip figure generation
```

## Output

- Console: model comparison metrics (χ², BIC, AIC) for all four variants
- Figures in `simulation/figs/`:
  - `geometry_hubble_comparison.png` — H(z) for all models + ΛCDM + data
  - `geometry_dm_fraction_comparison.png` — Ω_DM(z) evolution for 5D vs 4D
  - `geometry_growth_comparison.png` — fσ₈(z) for all models
  - `geometry_summary_comparison.png` — 6-panel summary
  - `geometry_weff_comparison.png` — effective w(z) comparison
- Results in `simulation/results/geometry_comparison.json`

## Key Testable Predictions

| Observable | 5D Spatial | 4D Temporal | ΛCDM |
|---|---|---|---|
| DM onset redshift | Sharp at z_contact | Gradual (quantum tunneling) | Always present |
| DM fraction at z=2 | Lower (not yet merged) | Higher (still interfering) | Same (~0.27) |
| Growth rate f(z) | Suppressed at low z | Enhanced at low z | Baseline |
| H(z) at z>1 | Higher DE → higher H | Lower DE → lower H | Smooth baseline |
| w_eff(z) | Crosses −1 from above | Crosses −1 from below | w = −1 constant |

The **data itself will discriminate** — especially high-redshift JWST+DESI observations.
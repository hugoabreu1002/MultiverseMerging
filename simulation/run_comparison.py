#!/usr/bin/env python3
"""
Merging Universes — 5D vs 4D Geometry Comparison.
Compares four model configurations:
1. 5D Spatial Brane (DE only)
2. 5D Spatial Brane + Foreign Dark Matter
3. 4D Temporal Quantum (DE only)
4. 4D Temporal Quantum + Foreign Dark Matter

Each is compared against ΛCDM baseline and the same observational data.
"""

import sys
import json
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from model import config
from model.merger_model import MergerModel
from data.jwst_queries import load_simulated_jwst_highz_data, load_pantheon_plus_data
from fitting.likelihood import total_log_likelihood, chi2_sne, bic, aic
from analysis.comparison_plots import (
    plot_geometry_comparison_hubble,
    plot_geometry_comparison_dm_fraction,
    plot_geometry_comparison_growth,
    plot_geometry_comparison_summary,
    plot_all_geometry_comparisons,
)


def run_model(params, label, include_dm=False):
    """Run a single model and return results."""
    p = config.MERGER_DEFAULTS.copy()
    p.update(params)
    p['include_foreign_dm'] = include_dm
    
    # Tag the label
    geometry = p.get('geometry', '5D')
    dm_tag = '_withDM' if include_dm else '_DEonly'
    run_label = f"{geometry}{dm_tag}"
    
    print(f"\n{'='*60}")
    print(f"  Running: {run_label}")
    print(f"{'='*60}")
    
    model = MergerModel(p)
    t0 = time.time()
    model.solve()
    model.solve_lcdm()
    elapsed = time.time() - t0
    print(f"  Solved in {elapsed:.2f}s")
    
    return model, run_label


def main():
    """Run 5D vs 4D geometry comparison."""
    print("=" * 70)
    print("  5D SPATIAL BRANE vs 4D TEMPORAL QUANTUM")
    print("  Geometry Comparison for Merging Universes")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    sne_data = load_pantheon_plus_data()
    jwst_data = load_simulated_jwst_highz_data()
    print(f"  SN Ia data: {len(sne_data['z'])} points")
    print(f"  JWST high-z: {len(jwst_data['z'])} galaxy mass function points")

    # --- Base parameters (shared across all models) ---
    base_params = {
        'alpha': 0.3,
        'beta': 2.5,
        'a_contact': 0.3,
        'N_universes': 2,
        'n_steps': 5000,
        'z_max': 10.0,
        'alpha_dm': 0.15,
        'beta_dm': 1.5,
        'smoothness': 10.0,
        'tunneling_scale': 0.15,
        'time_dilation_enhancement': 0.1,
        'interference_scale': 0.2,
        'overdensity_factor': 1.0,
    }

    # --- Run 5D Spatial Brane models ---
    params_5d = base_params.copy()
    params_5d['geometry'] = '5D'
    
    model_5d_de, label_5d_de = run_model(params_5d, '5D', include_dm=False)
    model_5d_full, label_5d_full = run_model(params_5d, '5D', include_dm=True)

    # --- Run 4D Temporal Quantum models ---
    params_4d = base_params.copy()
    params_4d['geometry'] = '4D'
    
    model_4d_de, label_4d_de = run_model(params_4d, '4D', include_dm=False)
    model_4d_full, label_4d_full = run_model(params_4d, '4D', include_dm=True)

    # --- Compute comparison metrics ---
    print("\n[4/5] Computing comparison metrics...")
    
    models = {
        label_5d_de: model_5d_de,
        label_5d_full: model_5d_full,
        label_4d_de: model_4d_de,
        label_4d_full: model_4d_full,
    }
    
    metrics = {}
    for label, model in models.items():
        z_sne = sne_data['z']
        mu_model = np.array([model.universe.distance_modulus(z) for z in z_sne])
        chi2 = chi2_sne(mu_model, sne_data['mu'], sne_data['mu_err'])
        
        # ΛCDM χ²
        if model.baseline_results is not None:
            mu_lcdm = np.interp(z_sne, model.baseline_results['z'],
                                model.baseline_results['mu'])
            chi2_lcdm = chi2_sne(mu_lcdm, sne_data['mu'], sne_data['mu_err'])
        else:
            chi2_lcdm = chi2
        
        n_data = len(z_sne)
        n_params = 4 if 'withDM' in label else 3  # extra params for DM
        
        metrics[label] = {
            'chi2': float(chi2),
            'chi2_lcdm': float(chi2_lcdm),
            'delta_chi2': float(chi2 - chi2_lcdm),
            'bic': float(bic(chi2, n_params, n_data)),
            'aic': float(aic(chi2, n_params)),
            'H0': model.results['H_km_s_Mpc'][0],
            'Omega_m': model.results['Omega_m'][0],
            'Omega_DE': model.results['Omega_DE'][0],
            'Omega_DM_foreign': model.results['Omega_DM_foreign'][0],
            'Omega_m_local': model.results['Omega_m_local'][0],
        }

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("  GEOMETRY COMPARISON RESULTS")
    print("=" * 70)
    
    header = f"{'Model':<25} {'χ²':<10} {'Δχ²':<10} {'BIC':<10} {'H₀':<10} {'Ω_m':<8}"
    print(f"\n  {header}")
    print(f"  {'-'*70}")
    for label, m in sorted(metrics.items()):
        row = f"{label:<25} {m['chi2']:<10.1f} {m['delta_chi2']:<+10.1f} {m['bic']:<10.1f} {m['H0']:<10.1f} {m['Omega_m']:<8.4f}"
        print(f"  {row}")
    
    print(f"\n  {'ΛCDM baseline':<25} {metrics[label_5d_de]['chi2_lcdm']:<10.1f} {'—':<10} {'—':<10} {config.PLANCK_2018['H0']:<10.1f} {config.PLANCK_2018['Omega_m']:<8.4f}")
    print(f"  {'-'*70}")
    
    # Determine best model
    best_model = min(metrics, key=lambda k: metrics[k]['bic'])
    print(f"\n  ** Best model (by BIC): {best_model} **")
    
    # Save metrics
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'geometry_comparison.json', 'w') as f:
        json.dump({
            'models': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                          for kk, vv in v.items()} 
                     for k, v in metrics.items()},
            'winner': best_model,
        }, f, indent=2)
    print(f"\n  Saved: geometry_comparison.json")

    # --- Generate comparison plots ---
    print("\n[5/5] Generating geometry comparison plots...")
    
    plot_all_geometry_comparisons(
        model_5d=model_5d_de,
        model_5d_dm=model_5d_full,
        model_4d=model_4d_de,
        model_4d_dm=model_4d_full,
        sne_data=sne_data,
        jwst_data=jwst_data,
        label_prefix='geometry_'
    )
    
    print("\n  All comparison plots saved to simulation/figs/")
    print("\nDone.")
    
    return metrics


if __name__ == '__main__':
    main()
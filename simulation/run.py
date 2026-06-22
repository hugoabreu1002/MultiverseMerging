#!/usr/bin/env python3
"""
Merging Universes Simulation — Main Entry Point.

A physically rigorous model of merging universes as an alternative to ΛCDM,
where dark energy is generated at the interface between overlapping universes.

Usage:
    python run.py                   # Run with default parameters
    python run.py --mcmc            # Run MCMC fitting
    python run.py --quick           # Quick test run
    python run.py --params '{"alpha": 0.01, "beta": 1.5, ...}'  # Custom params

Output:
    - Console: model comparison metrics, χ² values, BIC/AIC
    - Figures: simulation/figs/*.png
    - Data: simulation/results/*.npy
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
import time

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

from model import config
from model.merger_model import MergerModel
from data.jwst_queries import load_simulated_jwst_highz_data, load_pantheon_plus_data
from fitting.likelihood import total_log_likelihood, model_vs_lcdm_chi2, bic, aic
from fitting.mcmc import MergerMCMC
from analysis.plots import plot_all, plot_corner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merging Universes Model — ΛCDM Alternative'
    )
    parser.add_argument('--mcmc', action='store_true',
                       help='Run MCMC parameter estimation')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run (fewer steps, low resolution)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--params', type=str, default=None,
                       help='JSON string with custom parameters')
    parser.add_argument('--label', type=str, default='',
                       help='Label prefix for outputs')
    parser.add_argument('--no-lcdm', action='store_true',
                       help='Skip ΛCDM baseline comparison')
    return parser.parse_args()


def main():
    """Main simulation entry point."""
    args = parse_args()
    label = args.label or 'merger'

    print("=" * 70)
    print("  MERGING UNIVERSES COSMOLOGICAL MODEL")
    print("  An alternative to ΛCDM with interface-generated dark energy")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    sne_data = load_pantheon_plus_data()
    jwst_data = load_simulated_jwst_highz_data()
    print(f"  SN Ia data: {len(sne_data['z'])} points (simulated Pantheon+)")
    print(f"  JWST high-z: {len(jwst_data['z'])} galaxy mass function points")

    # --- Configure model parameters ---
    print("\n[2/5] Configuring model...")
    params = config.MERGER_DEFAULTS.copy()

    if args.quick:
        # Quick test parameters
        params['n_steps'] = 500
        params['z_max'] = 10.0
        params['alpha'] = 0.3
        params['beta'] = 2.5
        params['a_contact'] = 0.3
        print("  [QUICK MODE] Reduced resolution")
    elif args.params:
        # Custom JSON parameters
        custom = json.loads(args.params)
        params.update(custom)
        print(f"  Custom parameters: {custom}")

    print(f"  α (coupling) = {params['alpha']:.6e}")
    print(f"  β (amplification) = {params['beta']:.2f}")
    print(f"  d_init = {params['d_init']:.4f}")
    print(f"  v_init = {params['v_init']:.4f}")
    print(f"  N universes = {params['N_universes']}")
    print(f"  H0 = {params['H0_univ']:.1f} km/s/Mpc")
    print(f"  Ω_m = {params['Omega_m_univ']:.4f}")

    # --- Run MCMC if requested ---
    if args.mcmc:
        print("\n[3/5] Running MCMC parameter estimation...")
        n_steps = 500 if args.quick else 2000
        n_burn = 100 if args.quick else 500

        mcmc = MergerMCMC(
            sne_data=sne_data,
            jwst_data=jwst_data,
            free_params=['alpha', 'a_contact', 'beta'],
            n_steps=n_steps,
            n_burn=n_burn,
            n_walkers=32
        )
        t0 = time.time()
        result = mcmc.run(progress=True)
        elapsed = time.time() - t0
        print(f"  MCMC completed in {elapsed:.1f}s")

        mcmc.parameter_summary()
        model = mcmc.get_best_model()

        # Corner plot
        if not args.no_plots and mcmc.sampler is not None:
            try:
                plot_corner(mcmc.sampler, mcmc.free_params, label + '_')
            except Exception as e:
                print(f"  Corner plot failed: {e}")

    else:
        # --- Single run with current parameters ---
        print("\n[3/5] Solving Merger Model...")
        t0 = time.time()
        model = MergerModel(params)
        model.solve()
        elapsed = time.time() - t0
        print(f"  Merger model solved in {elapsed:.2f}s")

        if not args.no_lcdm:
            print("  Solving ΛCDM baseline for comparison...")
            model.solve_lcdm()

    # --- Compute likelihood and comparison metrics ---
    print("\n[4/5] Computing comparison metrics...")

    # Get model predictions for SN Ia data
    z_sne = sne_data['z']
    mu_model = np.array([model.universe.distance_modulus(z) for z in z_sne])

    # Compute χ²
    from fitting.likelihood import chi2_sne
    chi2_model = chi2_sne(mu_model, sne_data['mu'], sne_data['mu_err'])

    # ΛCDM χ²
    if model.baseline_results is not None:
        mu_lcdm = np.interp(z_sne, model.baseline_results['z'],
                            model.baseline_results['mu'])
        chi2_lcdm = chi2_sne(mu_lcdm, sne_data['mu'], sne_data['mu_err'])
    else:
        chi2_lcdm = chi2_model  # Fallback

    # BIC/AIC comparison
    n_data = len(z_sne)
    n_params_merger = 3  # alpha, beta, d_init
    n_params_lcdm = 2    # Omega_m, H0 (fixed in Planck)

    bic_merger = bic(chi2_model, n_params_merger, n_data)
    bic_lcdm = bic(chi2_lcdm, n_params_lcdm, n_data)
    aic_merger = aic(chi2_model, n_params_merger)
    aic_lcdm = aic(chi2_lcdm, n_params_lcdm)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'Merger Model':<18} {'ΛCDM':<18}")
    print(f"  {'-'*66}")
    print(f"  {'H0 [km/s/Mpc]':<30} {model.results['H_km_s_Mpc'][0]:<18.2f} {config.PLANCK_2018['H0']:<18.2f}")
    print(f"  {'Omega_m':<30} {model.results['Omega_m'][0]:<18.4f} {config.PLANCK_2018['Omega_m']:<18.4f}")

    if model.baseline_results:
        print(f"  {'Omega_DE':<30} {model.results['Omega_DE'][0]:<18.4f} {model.baseline_results['Omega_DE'][0]:<18.4f}")

    print(f"  {'chi2_SN':<30} {chi2_model:<18.2f} {chi2_lcdm:<18.2f}")
    print(f"  {'Delta chi2':<30} {chi2_model - chi2_lcdm:<+18.2f} {'N/A':<18}")
    print(f"  {'BIC':<30} {bic_merger:<18.2f} {bic_lcdm:<18.2f}")
    print(f"  {'AIC':<30} {aic_merger:<18.2f} {aic_lcdm:<18.2f}")

    # Hubble tension metric
    hubble_metric = model.hubble_tension_metric()
    print(f"\n  Hubble Tension:")
    print(f"  {'  Model H₀':<30} {hubble_metric['H0_model']:<.1f}")
    print(f"  {'  Planck 2018 H₀':<30} {hubble_metric['H0_Planck']:<.1f}")
    print(f"  {'  Riess 2022 H₀':<30} {hubble_metric['H0_Riess']:<.1f}")
    print(f"  {'  Δ vs Planck':<30} {hubble_metric['delta_vs_Planck']:<+.1f}")
    print(f"  {'  Δ vs Riess':<30} {hubble_metric['delta_vs_Riess']:<+.1f}")

    # Cosmic age at high z
    if not args.quick:
        z_test = [8, 10, 12, 14]
        print(f"\n  Cosmic Age at High Redshift:")
        print(f"  {'z':<8} {'Merger [Gyr]':<16} {'ΛCDM [Gyr]':<16} {'Δt [Myr]':<16}")
        for z_val in z_test:
            t_m = model.universe.t_of_z(z_val) / 3.15576e16
            t_l = 0
            if model.baseline_results is not None:
                t_l = np.interp(z_val, model.baseline_results['z'],
                               model.baseline_results['t']) / 3.15576e16
            dt_myr = (t_m - t_l) * 1000
            print(f"  {z_val:<8} {t_m:<16.3f} {t_l:<16.3f} {dt_myr:<+16.1f}")

    print("=" * 70)

    # --- Generate plots ---
    if not args.no_plots:
        print("\n[5/5] Generating plots...")
        try:
            plot_all(model, sne_data, jwst_data, label_prefix=label + '_')
            print(f"  Plots saved to {Path(__file__).parent / 'figs'}/")
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")
    else:
        print("\n[5/5] Plots skipped (--no-plots)")

    # --- Save results ---
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    summary = {
        'params': params,
        'chi2_model': float(chi2_model),
        'chi2_lcdm': float(chi2_lcdm),
        'bic_merger': float(bic_merger),
        'bic_lcdm': float(bic_lcdm),
        'aic_merger': float(aic_merger),
        'aic_lcdm': float(aic_lcdm),
        'H0_model': float(model.results['H_km_s_Mpc'][-1]),
        'Omega_m_model': float(model.results['Omega_m'][-1]),
        'Omega_DE_model': float(model.results['Omega_DE'][-1]),
        'hubble_tension': hubble_metric,
    }

    # Save as JSON
    output_path = results_dir / f'{label}_results.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save the full H(z) arrays for later analysis
    if model.results is not None:
        np.savez(results_dir / f'{label}_hubble.npz',
                 z=model.results['z'],
                 H=model.results['H_km_s_Mpc'],
                 H_lcdm=model.baseline_results['H_km_s_Mpc'] if model.baseline_results else None,
                 mu=model.results['mu'],
                 mu_lcdm=model.baseline_results['mu'] if model.baseline_results else None,
                 age=model.results['t'] / 3.15576e16,
                 age_lcdm=model.baseline_results['t'] / 3.15576e16 if model.baseline_results else None,
        )

    print("\nDone.")
    return summary


if __name__ == '__main__':
    main()
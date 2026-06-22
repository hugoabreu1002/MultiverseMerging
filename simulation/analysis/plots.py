"""
Plotting module for the merging universes simulation.
All figures are saved to simulation/figs/.

Produces:
1. H(z) comparison: merger model vs ΛCDM vs data
2. Distance modulus residuals: merger model vs ΛCDM vs Pantheon+
3. Dark energy density evolution: rho_DE(z) / rho_crit(z)
4. Effective w(z) from interface model
5. Growth rate fσ₈(z): merger vs ΛCDM
6. Cosmic age: t(z) -- merger vs ΛCDM
7. Galaxy mass function comparison with JWST high-z data
8. Brane separation and interface activation over time
9. Triangle/corner plot for MCMC chains (when available)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Figures directory
FIGS_DIR = Path(__file__).parent.parent / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name):
    """Save figure to figs/ directory in multiple formats."""
    path = FIGS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def plot_hubble_comparison(model, hz_data=None, label_prefix=''):
    """
    Plot H(z) comparison: merger model vs ΛCDM vs data points.

    Parameters
    ----------
    model : MergerModel (solved)
        Must have results and baseline_results populated.
    hz_data : dict, optional
        Observational H(z) data with 'z', 'H', 'H_err'
    label_prefix : str
        Prefix for saved filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    results = model.results
    baseline = model.baseline_results

    z = results['z']
    H_model = results['H_km_s_Mpc']
    H_lcdm = baseline['H_km_s_Mpc'] if baseline is not None else None

    # Main panel
    ax1.plot(z, H_model, 'b-', linewidth=2, label='Merger Model', alpha=0.8)
    if H_lcdm is not None:
        ax1.plot(z, H_lcdm, 'k--', linewidth=2, label='ΛCDM (Planck)', alpha=0.8)

    if hz_data is not None:
        ax1.errorbar(hz_data['z'], hz_data['H'], yerr=hz_data.get('H_err', None),
                     fmt='o', color='red', alpha=0.5, label='H(z) data',
                     markersize=4, capsize=2)

    ax1.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Residual panel
    if H_lcdm is not None:
        ax2.plot(z, H_model - H_lcdm, 'b-', linewidth=2, label='ΔH (Merger - ΛCDM)')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    if hz_data is not None:
        H_data_interp = np.interp(z, hz_data['z'], hz_data['H'])
        ax2.plot(z, H_model - H_data_interp, 'g-', linewidth=1.5,
                 label='ΔH (Merger - Data)')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('ΔH [km/s/Mpc]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Hubble Parameter H(z)', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}hubble_comparison.png')
    plt.close(fig)


def plot_distance_modulus(model, sne_data=None, label_prefix=''):
    """
    Plot distance modulus μ(z) with SN Ia data and residuals.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    results = model.results
    baseline = model.baseline_results

    z = results['z']
    mu_model = results['mu']
    mu_lcdm = baseline['mu'] if baseline is not None else None

    # Main panel
    ax1.plot(z, mu_model, 'b-', linewidth=2, label='Merger Model', alpha=0.8)
    if mu_lcdm is not None:
        ax1.plot(z, mu_lcdm, 'k--', linewidth=2, label='ΛCDM', alpha=0.8)

    if sne_data is not None:
        ax1.errorbar(sne_data['z'], sne_data['mu'], yerr=sne_data['mu_err'],
                     fmt='o', color='red', alpha=0.3, label='Pantheon+ (sim)',
                     markersize=2, capsize=1)

    ax1.set_ylabel('μ(z) [mag]', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Residuals w.r.t. ΛCDM
    if mu_lcdm is not None:
        residual = mu_model - mu_lcdm
        ax2.plot(z, residual, 'b-', linewidth=2, label='Δμ (Merger - ΛCDM)')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        if sne_data is not None:
            mu_data_interp = np.interp(z, sne_data['z'], sne_data['mu'])
            ax2.plot(z, mu_model - mu_data_interp, 'g-', linewidth=1.5,
                     label='Δμ (Merger - Data)')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Δμ [mag]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Distance Modulus μ(z)', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}distance_modulus.png')
    plt.close(fig)


def plot_dark_energy_evolution(model, label_prefix=''):
    """
    Plot dark energy density evolution and fraction.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    results = model.results
    baseline = model.baseline_results

    z = results['z']

    # Panel 1: ρ_DE / ρ_crit
    rho_DE_frac = results['rho_DE'] / (
        3.0 * results['H']**2 / (8.0 * np.pi * 6.67430e-11)
    )

    ax1.plot(z, rho_DE_frac, 'b-', linewidth=2, label='Merger ρ_DE/ρ_crit')
    if baseline is not None:
        rho_L_frac = baseline['rho_DE'] / (
            3.0 * baseline['H']**2 / (8.0 * np.pi * 6.67430e-11)
        )
        ax1.axhline(y=0.685, color='k', linestyle='--', linewidth=2,
                    label='ΛCDM Ω_Λ = 0.685')
    ax1.set_ylabel('Ω_DE(z)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)

    # Panel 2: Ω_m
    ax2.plot(z, results['Omega_m'], 'r-', linewidth=2, label='Merger Ω_m')
    if baseline is not None:
        ax2.plot(z, baseline['Omega_m'], 'k--', linewidth=2, label='ΛCDM Ω_m')
    ax2.set_ylabel('Ω_m(z)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)

    # Panel 3: Density ratio ρ_DE / ρ_m
    rho_ratio = results['rho_DE'] / (results['rho_m'] + 1e-30)
    ax3.plot(z, rho_ratio, 'b-', linewidth=2, label='ρ_DE / ρ_m (Merger)')
    if baseline is not None:
        rho_ratio_lcdm = baseline['rho_DE'] / (baseline['rho_m'] + 1e-30)
        ax3.plot(z, rho_ratio_lcdm, 'k--', linewidth=2, label='ρ_DE / ρ_m (ΛCDM)')
    ax3.set_xlabel('Redshift z', fontsize=12)
    ax3.set_ylabel('ρ_DE / ρ_m', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    ax3.set_yscale('log')

    plt.suptitle('Dark Energy Evolution', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}dark_energy_evolution.png')
    plt.close(fig)


def plot_growth_comparison(model, growth_data=None, label_prefix=''):
    """
    Plot growth factor D(z) and growth rate fσ₈(z).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    results = model.results
    baseline = model.baseline_results

    z = results['z']

    # Growth factor
    ax1.plot(z, results['D'], 'b-', linewidth=2, label='Merger')
    if baseline is not None:
        ax1.plot(z, baseline['D'], 'k--', linewidth=2, label='ΛCDM')
    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('D(z)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # fσ₈
    sigma8_0 = 0.811
    fsigma8_model = results['f'] * sigma8_0 * results['D']
    ax2.plot(z, fsigma8_model, 'b-', linewidth=2, label='Merger')

    if baseline is not None:
        fsigma8_lcdm = baseline['f'] * sigma8_0 * baseline['D']
        ax2.plot(z, fsigma8_lcdm, 'k--', linewidth=2, label='ΛCDM')

    if growth_data is not None:
        ax2.errorbar(growth_data['z'], growth_data['fsigma8'],
                     yerr=growth_data['fsigma8_err'],
                     fmt='o', color='red', alpha=0.5,
                     label='fσ₈ data', markersize=4, capsize=2)

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('fσ₈(z)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)

    plt.suptitle('Growth of Structure', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}growth_comparison.png')
    plt.close(fig)


def plot_cosmic_age(model, label_prefix=''):
    """
    Plot cosmic age t(z) for merger model vs ΛCDM.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    results = model.results
    baseline = model.baseline_results

    z = results['z']
    t_model = results['t'] / (3.15576e16)  # Convert s to Gyr
    t_lcdm = baseline['t'] / 3.15576e16 if baseline is not None else None

    ax1.plot(z, t_model, 'b-', linewidth=2, label='Merger Model')
    if t_lcdm is not None:
        ax1.plot(z, t_lcdm, 'k--', linewidth=2, label='ΛCDM')
    ax1.set_ylabel('Age [Gyr]', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Age difference
    if t_lcdm is not None:
        dt = t_model - t_lcdm
        ax2.plot(z, dt, 'b-', linewidth=2, label='Δt (Merger - ΛCDM)')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # Highlight high-z region (JWST interest)
        ax2.axvspan(8, 15, alpha=0.1, color='orange', label='JWST high-z')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Δt [Gyr]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Cosmic Age', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}cosmic_age.png')
    plt.close(fig)


def plot_brane_evolution(model, label_prefix=''):
    """
    Plot brane separation and interface activation over cosmic time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    brane_system = model.brane_system
    history = brane_system.history

    if not history['t']:
        print("  No brane history available. Run BraneSystem.step() first.")
        return

    t_hist = history['t']
    d_hist = history['d']

    # Scale factor vs brane separation
    results = model.results
    z = results['z']
    a = results['a']

    # Compute brane separation as function of redshift
    # (Map brane time to cosmic time)
    ax1.plot(t_hist, d_hist, 'b-', linewidth=2)
    ax1.set_xlabel('Time [arbitrary units]', fontsize=12)
    ax1.set_ylabel('Brane Separation d', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2.0, color='k', linestyle='--', alpha=0.5,
                label='Contact (d = 2r)')
    ax1.legend(fontsize=10)

    # Interface activation
    activation = history['activation']
    ax2.plot(t_hist, activation, 'r-', linewidth=2)
    ax2.set_xlabel('Time [arbitrary units]', fontsize=12)
    ax2.set_ylabel('Interface Activation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.suptitle('Brane Dynamics', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}brane_evolution.png')
    plt.close(fig)


def plot_jwst_mass_function(model, jwst_data, label_prefix=''):
    """
    Plot JWST high-z galaxy mass function comparison.
    Shows how merger model provides more cosmic time for early galaxy formation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    z_jwst = np.array(jwst_data['z'])
    logM_jwst = np.array(jwst_data['logM'])
    log_phi_data = np.array(jwst_data['log_phi'])

    # Model cosmic time at each z
    z_unique = np.sort(np.unique(z_jwst))

    t_model = np.array([model.universe.t_of_z(z) for z in z_unique])
    t_lcdm = np.array([
        _lcdm_age_simple(z, 0.315)
        for z in z_unique
    ])
    t_model_gyr = t_model / 3.15576e16
    t_lcdm_gyr = t_lcdm / 3.15576e16

    # Panel 1: Cosmic age at high z
    ax1.plot(z_unique, t_model_gyr, 'b-o', linewidth=2, label='Merger Model')
    ax1.plot(z_unique, t_lcdm_gyr, 'k--o', linewidth=2, label='ΛCDM')
    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('Cosmic Age [Gyr]', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(5, 16)

    # Panel 2: Mass function enhancement
    # Enhancement factor = t_model / t_lcdm
    enhancement = t_model / t_lcdm

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(z_unique)))
    for i, z_val in enumerate(z_unique):
        mask = z_jwst == z_val
        ax2.scatter(logM_jwst[mask], log_phi_data[mask],
                   color=colors[i], s=60, label=f'z={z_val:.1f} (obs)',
                   alpha=0.8, edgecolors='k', linewidth=0.5)
        # Model: enhanced mass function
        log_phi_model = log_phi_data[mask] + np.log10(enhancement[i])
        ax2.plot(logM_jwst[mask], log_phi_model, color=colors[i],
                linestyle='-', marker='x', linewidth=2, markersize=8,
                label=f'z={z_val:.1f} (model)')

    ax2.set_xlabel('log₁₀(M*/M☉)', fontsize=12)
    ax2.set_ylabel('log₁₀(φ) [Mpc⁻³ dex⁻¹]', fontsize=12)
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('JWST High-z Galaxy Mass Function', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}jwst_mass_function.png')
    plt.close(fig)


def plot_summary_panel(model, sne_data=None, jwst_data=None, label_prefix=''):
    """
    Create a comprehensive summary figure with key results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    results = model.results
    baseline = model.baseline_results
    z = results['z']

    # 1. H(z)
    ax = axes[0, 0]
    ax.plot(z, results['H_km_s_Mpc'], 'b-', linewidth=2, label='Merger')
    if baseline is not None:
        ax.plot(z, baseline['H_km_s_Mpc'], 'k--', linewidth=2, label='ΛCDM')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('H(z) [km/s/Mpc]', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_title('Hubble Parameter', fontsize=10)

    # 2. Distance modulus
    ax = axes[0, 1]
    ax.plot(z, results['mu'], 'b-', linewidth=2, label='Merger')
    if baseline is not None:
        ax.plot(z, baseline['mu'], 'k--', linewidth=2, label='ΛCDM')
    if sne_data is not None:
        ax.errorbar(sne_data['z'], sne_data['mu'], yerr=sne_data['mu_err'],
                   fmt='o', color='red', alpha=0.2, markersize=1, capsize=0)
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('μ(z)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    ax.set_title('Distance Modulus', fontsize=10)

    # 3. Ω_DE and Ω_m
    ax = axes[0, 2]
    ax.plot(z, results['Omega_DE'], 'b-', linewidth=2, label='Ω_DE (Merger)')
    ax.plot(z, results['Omega_m'], 'r-', linewidth=2, label='Ω_m (Merger)')
    if baseline is not None:
        ax.axhline(y=0.685, color='k', linestyle='--', linewidth=1, label='Ω_Λ (ΛCDM)')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('Ω(z)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_title('Density Parameters', fontsize=10)

    # 4. Growth fσ₈
    ax = axes[1, 0]
    sigma8_0 = 0.811
    fsigma8 = results['f'] * sigma8_0 * results['D']
    ax.plot(z, fsigma8, 'b-', linewidth=2, label='Merger')
    if baseline is not None:
        fsigma8_lcdm = baseline['f'] * sigma8_0 * baseline['D']
        ax.plot(z, fsigma8_lcdm, 'k--', linewidth=2, label='ΛCDM')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('fσ₈(z)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_title('Growth Rate', fontsize=10)

    # 5. Cosmic age
    ax = axes[1, 1]
    t_model_gyr = results['t'] / 3.15576e16
    ax.plot(z, t_model_gyr, 'b-', linewidth=2, label='Merger')
    if baseline is not None:
        t_lcdm_gyr = baseline['t'] / 3.15576e16
        ax.plot(z, t_lcdm_gyr, 'k--', linewidth=2, label='ΛCDM')
    ax.axvspan(8, 15, alpha=0.1, color='orange', label='JWST high-z')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('Age [Gyr]', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_title('Cosmic Age', fontsize=10)

    # 6. Δχ² summary
    ax = axes[1, 2]
    ax.text(0.5, 0.7, f"H₀ = {results['H_km_s_Mpc'][-1]:.1f} km/s/Mpc",
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.5, 0.5, f"Ω_m = {results['Omega_m'][-1]:.3f}",
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.5, 0.3, f"Ω_DE = {results['Omega_DE'][-1]:.3f}",
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.5, 0.1, f"Label: {model.label}",
            transform=ax.transAxes, fontsize=10, ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Model Parameters', fontsize=10)

    plt.tight_layout()
    savefig(fig, f'{label_prefix}summary_panel.png')
    plt.close(fig)


def plot_corner(sampler, free_params, label_prefix=''):
    """
    Corner plot of MCMC chains (if corner package is available).
    """
    try:
        import corner
        chain = sampler.get_chain(discard=500, flat=True)
        fig = corner.corner(chain, labels=free_params, show_titles=True,
                           title_fmt='.4f', quantiles=[0.16, 0.5, 0.84])
        savefig(fig, f'{label_prefix}mcmc_corner.png')
        plt.close(fig)
    except ImportError:
        print("  corner package not installed. Skipping corner plot.")


def _lcdm_age_simple(z, Omega_m):
    """
    Simple ΛCDM age computation for comparison.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model import config
    Omega_L = 1.0 - Omega_m - 9.2e-5
    H0 = 67.4 * config.km_s_Mpc_to_1_per_s
    z_max = 2000
    z_grid = np.linspace(z, z_max, 100000)
    E = np.sqrt(Omega_m * (1 + z_grid)**3 + 9.2e-5 * (1 + z_grid)**4 + Omega_L)
    from scipy.integrate import trapezoid
    integral = trapezoid(1.0 / ((1 + z_grid) * E), z_grid)
    return integral / H0


def plot_all(model, sne_data=None, jwst_data=None, hz_data=None,
             growth_data=None, label_prefix=''):
    """
    Generate all standard plots.
    """
    print("Generating plots...")
    plot_hubble_comparison(model, hz_data, label_prefix)
    plot_distance_modulus(model, sne_data, label_prefix)
    plot_dark_energy_evolution(model, label_prefix)
    plot_growth_comparison(model, growth_data, label_prefix)
    plot_cosmic_age(model, label_prefix)
    plot_summary_panel(model, sne_data, jwst_data, label_prefix)

    if jwst_data is not None:
        plot_jwst_mass_function(model, jwst_data, label_prefix)


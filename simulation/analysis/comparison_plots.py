"""
Comparison plots for 5D Spatial Brane vs 4D Temporal Quantum geometry.
Generates side-by-side and combined figures showing:
1. H(z) comparison: all four models + ΛCDM + data
2. Foreign DM fraction Ω_DM(z) evolution
3. Growth rate fσ₈(z) comparison
4. Summary panel with model parameters and fit metrics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erfc

FIGS_DIR = Path(__file__).parent.parent / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name):
    """Save figure to figs/ directory."""
    path = FIGS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_geometry_comparison_hubble(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                     hz_data=None, label_prefix=''):
    """
    Plot H(z) for all four model variants + ΛCDM.
    Top panel: H(z) curves
    Bottom panel: residuals relative to ΛCDM
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # ΛCDM baseline
    baseline = model_5d.baseline_results
    z_lcdm = baseline['z']
    H_lcdm = baseline['H_km_s_Mpc']

    # Model definitions with colors and styles
    configs = [
        (model_5d, '5D Spatial (DE only)', 'C0', '-'),
        (model_5d_dm, '5D Spatial + DM', 'C1', '--'),
        (model_4d, '4D Temporal (DE only)', 'C2', '-'),
        (model_4d_dm, '4D Temporal + DM', 'C3', '--'),
    ]

    # Main panel
    ax1.plot(z_lcdm, H_lcdm, 'k-', linewidth=2, label='ΛCDM (Planck)', alpha=0.7)

    for model, label, color, style in configs:
        z = model.results['z']
        H = model.results['H_km_s_Mpc']
        ax1.plot(z, H, color=color, linestyle=style, linewidth=2,
                label=label, alpha=0.8)

    if hz_data is not None:
        ax1.errorbar(hz_data['z'], hz_data['H'], yerr=hz_data.get('H_err', None),
                     fmt='o', color='gray', alpha=0.4, label='H(z) data',
                     markersize=3, capsize=2)

    ax1.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_title('Hubble Parameter — Geometry Comparison', fontsize=14)

    # Residual panel: ΔH = model - ΛCDM
    for model, label, color, style in configs:
        z = model.results['z']
        H = model.results['H_km_s_Mpc']
        H_lcdm_interp = np.interp(z, z_lcdm, H_lcdm)
        residual = H - H_lcdm_interp
        ax2.plot(z, residual, color=color, linestyle=style, linewidth=1.5,
                label=label, alpha=0.8)

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('ΔH [km/s/Mpc] (vs ΛCDM)', fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)

    plt.tight_layout()
    savefig(fig, f'{label_prefix}hubble_comparison.png')


def plot_geometry_comparison_dm_fraction(model_5d, model_5d_dm,
                                          model_4d, model_4d_dm,
                                          label_prefix=''):
    """
    Plot the foreign dark matter fraction Ω_DM_foreign(z) for both geometries.
    Also shows the total matter density split.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data from the models with DM
    for ax_row, (model_de, model_dm, geo_label, color_de, color_dm) in zip(
        axes,
        [
            (model_5d, model_5d_dm, '5D Spatial', 'C0', 'C1'),
            (model_4d, model_4d_dm, '4D Temporal', 'C2', 'C3'),
        ]
    ):
        z = model_dm.results['z']
        a = model_dm.results['a']
        
        # Panels: [0] = DM fraction, [1] = matter density components
        
        # Panel 0: Ω_DM_foreign(z) evolution
        ax = ax_row[0]
        ax.plot(z, model_dm.results['Omega_DM_foreign'], color=color_dm,
               linewidth=2, label=f'{geo_label} Ω_DM (foreign)', alpha=0.9)
        ax.plot(z, model_dm.results['Omega_m_local'], color=color_de,
               linewidth=2, linestyle='--', label=f'{geo_label} Ω_m (local)',
               alpha=0.9)
        ax.plot(z, model_dm.results['Omega_m'], color=color_dm,
               linewidth=2, linestyle=':', label=f'{geo_label} Ω_m (total)',
               alpha=0.7)
        
        # Mark the DM onset
        a_contact = model_dm.params.get('a_contact', 0.3)
        z_contact = 1.0 / a_contact - 1.0
        ax.axvline(x=z_contact, color='gray', linestyle=':', alpha=0.5,
                  label=f'Contact z={z_contact:.1f}')
        
        ax.set_xlabel('Redshift z', fontsize=11)
        ax.set_ylabel('Ω(z)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)
        ax.set_title(f'{geo_label}: Density Evolution', fontsize=12)

        # Panel 1: ρ_DM_foreign / ρ_m_local (enhancement ratio)
        ax = ax_row[1]
        ratio = model_dm.results['rho_DM_foreign'] / (model_dm.results['rho_m_local'] + 1e-30)
        ax.plot(z, ratio, color=color_dm, linewidth=2,
               label=f'{geo_label} ρ_DM/ρ_m(local)')
        
        # Also show growth factor overdensity
        D = model_dm.results['D']
        overdensity = D / D[-1]  # Normalized growth
        ax.plot(z, overdensity, color='gray', linewidth=1.5, linestyle='--',
               label='Growth D(z) (normalized)', alpha=0.7)
        
        ax.set_xlabel('Redshift z', fontsize=11)
        ax.set_ylabel('Density Ratio', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)
        ax.set_yscale('log')
        ax.set_title(f'{geo_label}: DM Enhancement', fontsize=12)

    plt.suptitle('Foreign Dark Matter from Merging Universes', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}dm_fraction_comparison.png')


def plot_geometry_comparison_growth(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                     growth_data=None, label_prefix=''):
    """
    Plot growth factor D(z) and fσ₈(z) for all models.
    Foreign DM changes growth via modified expansion history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    baseline = model_5d.baseline_results

    configs = [
        (model_5d, '5D (DE only)', 'C0', '-'),
        (model_5d_dm, '5D + DM', 'C1', '--'),
        (model_4d, '4D (DE only)', 'C2', '-'),
        (model_4d_dm, '4D + DM', 'C3', '--'),
    ]

    # Growth factor
    z_lcdm = baseline['z']
    ax1.plot(z_lcdm, baseline['D'], 'k-', linewidth=2, label='ΛCDM', alpha=0.7)

    for model, label, color, style in configs:
        z = model.results['z']
        ax1.plot(z, model.results['D'], color=color, linestyle=style,
                linewidth=2, label=label, alpha=0.8)

    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('Growth Factor D(z)', fontsize=12)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_title('Growth Factor', fontsize=12)

    # fσ₈
    sigma8_0 = 0.811
    ax2.plot(z_lcdm, baseline['f'] * sigma8_0 * baseline['D'],
            'k-', linewidth=2, label='ΛCDM', alpha=0.7)

    for model, label, color, style in configs:
        z = model.results['z']
        fsigma8 = model.results['f'] * sigma8_0 * model.results['D']
        ax2.plot(z, fsigma8, color=color, linestyle=style,
                linewidth=2, label=label, alpha=0.8)

    if growth_data is not None:
        ax2.errorbar(growth_data['z'], growth_data['fsigma8'],
                    yerr=growth_data.get('fsigma8_err', None),
                    fmt='o', color='gray', alpha=0.4,
                    label='fσ₈ data', markersize=4, capsize=2)

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('fσ₈(z)', fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)
    ax2.set_title('Growth Rate fσ₈(z)', fontsize=12)

    plt.suptitle('Growth of Structure — Geometry Comparison', fontsize=14)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}growth_comparison.png')


def plot_geometry_comparison_summary(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                      sne_data=None, label_prefix=''):
    """
    Summary comparison figure with 6 panels showing key discriminants.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    baseline = model_5d.baseline_results

    configs = [
        (model_5d, '5D (DE only)', 'C0', '-'),
        (model_5d_dm, '5D + DM', 'C1', '--'),
        (model_4d, '4D (DE only)', 'C2', '-'),
        (model_4d_dm, '4D + DM', 'C3', '--'),
    ]

    # 1. H(z) at low z (inset for Hubble tension)
    ax = axes[0, 0]
    z_lcdm = baseline['z']
    ax.plot(z_lcdm, baseline['H_km_s_Mpc'], 'k-', linewidth=2, label='ΛCDM', alpha=0.7)
    for model, label, color, style in configs:
        ax.plot(model.results['z'], model.results['H_km_s_Mpc'],
               color=color, linestyle=style, linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('H(z) [km/s/Mpc]', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_title('Hubble Parameter', fontsize=11)

    # 2. Distance modulus residuals vs ΛCDM
    ax = axes[0, 1]
    z_lcdm = baseline['z']
    mu_lcdm = baseline['mu']
    for model, label, color, style in configs:
        z = model.results['z']
        mu = model.results['mu']
        mu_lcdm_interp = np.interp(z, z_lcdm, mu_lcdm)
        ax.plot(z, mu - mu_lcdm_interp, color=color, linestyle=style,
               linewidth=1.5, label=label, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('Δμ [mag] (vs ΛCDM)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    ax.set_title('Distance Modulus Residual', fontsize=11)

    # 3. Foreign DM fraction
    ax = axes[0, 2]
    for model, label, color, style in configs:
        if hasattr(model.results, 'get') and 'Omega_DM_foreign' in model.results:
            omega_dm = model.results['Omega_DM_foreign']
            if np.max(omega_dm) > 1e-10:  # Only plot if non-zero
                ax.plot(model.results['z'], omega_dm, color=color,
                       linestyle=style, linewidth=2, label=label, alpha=0.8)
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('Ω_DM (foreign)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_title('Foreign Dark Matter Fraction', fontsize=11)

    # 4. fσ₈ growth rate
    ax = axes[1, 0]
    sigma8_0 = 0.811
    for model, label, color, style in configs:
        fsigma8 = model.results['f'] * sigma8_0 * model.results['D']
        ax.plot(model.results['z'], fsigma8, color=color, linestyle=style,
               linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('fσ₈(z)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_title('Growth Rate fσ₈(z)', fontsize=11)

    # 5. Cosmic age at high z (JWST relevance)
    ax = axes[1, 1]
    for model, label, color, style in configs:
        t_gyr = model.results['t'] / 3.15576e16
        ax.plot(model.results['z'], t_gyr, color=color, linestyle=style,
               linewidth=1.5, label=label, alpha=0.8)
    ax.axvspan(8, 15, alpha=0.1, color='orange', label='JWST high-z')
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('Cosmic Age [Gyr]', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_title('Cosmic Age', fontsize=11)

    # 6. Effective DE equation of state w(z)
    ax = axes[1, 2]
    for model, label, color, style in configs:
        z = model.results['z']
        H = model.results['H']
        H0 = model.results['H_km_s_Mpc'][0] * 1e3 / 3.085677581e22  # SI
        Omega_m0 = model.results['Omega_m'][0]  # This is Omega_m at z=0
        
        # w_eff(z) = -1 + (1+z)/3 * dln(H^2)/dz / (1 - Omega_m(z))
        # Simplified: w_eff = (2/3 * (1+z) * H'/H - 1) / (1 - Omega_m)
        dH_dz = np.gradient(H, z)
        factor = (2.0/3.0) * (1.0 + z) * dH_dz / (H + 1e-30) - 1.0
        Omega_m_z = model.results['Omega_m']
        w_eff = factor / ((1.0 - Omega_m_z) + 1e-30)
        w_eff = np.clip(w_eff, -3, 1)  # Clip unphysical values
        
        ax.plot(z, w_eff, color=color, linestyle=style,
               linewidth=1.5, label=label, alpha=0.8)
    ax.axhline(y=-1, color='k', linestyle=':', linewidth=1, alpha=0.7, label='ΛCDM w=−1')
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('w_eff(z)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(-2, 0.5)
    ax.set_title('Effective w(z)', fontsize=11)

    plt.suptitle('5D Spatial vs 4D Temporal — Model Comparison', fontsize=16)
    plt.tight_layout()
    savefig(fig, f'{label_prefix}summary_comparison.png')


def plot_geometry_comparison_weff(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                   label_prefix=''):
    """
    Detailed effective equation of state w(z) comparison.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    baseline = model_5d.baseline_results
    
    configs = [
        (model_5d, '5D (DE only)', 'C0', '-'),
        (model_5d_dm, '5D + DM', 'C1', '--'),
        (model_4d, '4D (DE only)', 'C2', '-'),
        (model_4d_dm, '4D + DM', 'C3', '--'),
    ]

    for model, label, color, style in configs:
        z = model.results['z']
        H = model.results['H']
        dH_dz = np.gradient(H, z)
        Omega_m_z = model.results['Omega_m']
        
        # w_eff(z) from the Hubble parameter
        factor = (2.0/3.0) * (1.0 + z) * dH_dz / (H + 1e-30) - 1.0
        w_eff = factor / ((1.0 - Omega_m_z) + 1e-30)
        w_eff = np.clip(w_eff, -3, 1)
        
        ax.plot(z, w_eff, color=color, linestyle=style,
               linewidth=2, label=label, alpha=0.8)

    ax.axhline(y=-1, color='k', linestyle=':', linewidth=2, alpha=0.7, label='ΛCDM w=−1')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-1/3, color='gray', linestyle=':', linewidth=0.5, alpha=0.5,
              label='w=−1/3 (no accel.)')
    
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Effective Equation of State w_eff(z)', fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(-2.5, 0.5)
    ax.set_title('Effective Dark Energy Equation of State', fontsize=14)

    plt.tight_layout()
    savefig(fig, f'{label_prefix}weff_comparison.png')


def plot_all_geometry_comparisons(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                   sne_data=None, jwst_data=None, hz_data=None,
                                   growth_data=None, label_prefix=''):
    """
    Generate all geometry comparison plots.
    """
    print("  Hubble comparison...")
    plot_geometry_comparison_hubble(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                     hz_data, label_prefix)

    print("  DM fraction comparison...")
    plot_geometry_comparison_dm_fraction(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                          label_prefix)

    print("  Growth comparison...")
    plot_geometry_comparison_growth(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                     growth_data, label_prefix)

    print("  Summary panel...")
    plot_geometry_comparison_summary(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                      sne_data, label_prefix)

    print("  w(z) comparison...")
    plot_geometry_comparison_weff(model_5d, model_5d_dm, model_4d, model_4d_dm,
                                   label_prefix)

    print("  All comparison plots generated.")
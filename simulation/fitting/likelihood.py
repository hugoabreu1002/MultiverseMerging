"""
Likelihood functions for model comparison against data.
Computes χ² and log-likelihood for:
1. SN Ia distance modulus (Pantheon+)
2. High-z galaxy mass function (JWST)
3. Hubble parameter H(z) measurements
4. Growth rate fσ₈(z) measurements
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats
from model import config
from model.merger_model import MergerModel


def chi2_sne(mu_model, mu_data, mu_err):
    """
    χ² for SN Ia distance modulus.

    Parameters
    ----------
    mu_model : array-like
        Model-predicted distance moduli
    mu_data : array-like
        Observed distance moduli
    mu_err : array-like
        Uncertainties on observed moduli

    Returns
    -------
    chi2 : float
    """
    residual = mu_model - mu_data
    return np.sum((residual / mu_err)**2)


def chi2_hubble(H_model, H_data, H_err):
    """
    χ² for H(z) measurements (cosmic chronometers).

    Parameters
    ----------
    H_model : array-like
        Model-predicted H(z) in km/s/Mpc
    H_data : array-like
        Observed H(z) in km/s/Mpc
    H_err : array-like
        Uncertainties on observed H(z)

    Returns
    -------
    chi2 : float
    """
    residual = H_model - H_data
    return np.sum((residual / H_err)**2)


def chi2_growth(fsigma8_model, fsigma8_data, fsigma8_err):
    """
    χ² for growth rate fσ₈(z) measurements.

    Parameters
    ----------
    fsigma8_model : array-like
        Model-predicted fσ₈(z)
    fsigma8_data : array-like
        Observed fσ₈(z)
    fsigma8_err : array-like
        Uncertainties on observed fσ₈(z)

    Returns
    -------
    chi2 : float
    """
    residual = fsigma8_model - fsigma8_data
    return np.sum((residual / fsigma8_err)**2)


def chi2_jwst_mass_function(log_phi_model, log_phi_data, log_phi_err):
    """
    χ² for JWST high-z galaxy mass function.

    Parameters
    ----------
    log_phi_model : array-like
        Model-predicted log10(number density / Mpc^-3 dex^-1)
    log_phi_data : array-like
        Observed values
    log_phi_err : array-like
        Uncertainties

    Returns
    -------
    chi2 : float
    """
    residual = log_phi_model - log_phi_data
    return np.sum((residual / log_phi_err)**2)


def total_log_likelihood(model, sne_data, jwst_data=None, hz_data=None,
                          growth_data=None, params=None):
    """
    Compute total log-likelihood for the merger model.

    Parameters
    ----------
    model : MergerModel instance (already solved)
    sne_data : dict
        SN Ia data with 'z', 'mu', 'mu_err'
    jwst_data : dict, optional
        High-z galaxy data with 'z', 'logM', 'log_phi'
    hz_data : dict, optional
        H(z) data with 'z', 'H', 'H_err'
    growth_data : dict, optional
        fσ₈ data with 'z', 'fsigma8', 'fsigma8_err'
    params : dict, optional
        Extra parameters

    Returns
    -------
    log_like : float
        Total log-likelihood
    components : dict
        Individual χ² components
    """
    chi2_total = 0.0
    components = {}

    # --- SN Ia likelihood ---
    if sne_data is not None:
        z_sne = sne_data['z']
        mu_model = np.array([model.universe.distance_modulus(z) for z in z_sne])
        chi2_sne_val = chi2_sne(mu_model, sne_data['mu'], sne_data['mu_err'])
        chi2_total += chi2_sne_val
        components['chi2_sne'] = chi2_sne_val
        components['ndata_sne'] = len(z_sne)

    # --- JWST high-z galaxy likelihood ---
    if jwst_data is not None:
        # Compute the cosmic time available for galaxy formation at each z
        # Model prediction: earlier interface DE gives more time at fixed z
        # Higher mass function → more time available
        z_jwst = np.array(jwst_data['z'])
        logM_jwst = np.array(jwst_data['logM'])

        # Get cosmic age at each z from model
        t_model = np.array([model.universe.t_of_z(z) for z in z_jwst])

        # ΛCDM baseline age for comparison
        lcdm_t = np.array([
            _lcdm_age(z, config.PLANCK_2018['Omega_m'],
                      config.PLANCK_2018['H0'])
            for z in z_jwst
        ])

        # Model prediction: more time → higher mass function
        # Simple phenomenological model: log_phi ∝ log(t_model / t_lcdm)
        t_ratio = t_model / (lcdm_t + 1e-30)
        log_phi_model = np.array(jwst_data['log_phi']) + np.log10(t_ratio)

        log_phi_data = np.array(jwst_data['log_phi'])
        log_phi_err = 0.3 * np.ones_like(log_phi_data)  # Approximate

        chi2_jwst_val = chi2_jwst_mass_function(
            log_phi_model, log_phi_data, log_phi_err
        )
        chi2_total += chi2_jwst_val
        components['chi2_jwst'] = chi2_jwst_val
        components['ndata_jwst'] = len(z_jwst)

    # --- H(z) likelihood ---
    if hz_data is not None:
        z_hz = hz_data['z']
        H_model = model.universe.H_of_z(z_hz) / config.km_s_Mpc_to_1_per_s
        chi2_hz_val = chi2_hubble(H_model, hz_data['H'], hz_data['H_err'])
        chi2_total += chi2_hz_val
        components['chi2_hz'] = chi2_hz_val
        components['ndata_hz'] = len(z_hz)

    # --- Growth likelihood ---
    if growth_data is not None:
        z_g = growth_data['z']
        D = model.universe.growth_factor(z_g)
        f = model.universe.growth_rate(z_g)
        sigma8_0 = config.PLANCK_2018['sigma8']
        fsigma8_model = f * sigma8_0 * D
        chi2_g_val = chi2_growth(
            fsigma8_model, growth_data['fsigma8'], growth_data['fsigma8_err']
        )
        chi2_total += chi2_g_val
        components['chi2_growth'] = chi2_g_val
        components['ndata_growth'] = len(z_g)

    # Log-likelihood: ln L = -χ²/2
    log_like = -0.5 * chi2_total
    components['chi2_total'] = chi2_total
    components['log_likelihood'] = log_like

    return log_like, components


def _lcdm_age(z, Omega_m, H0):
    """
    Cosmic age at redshift z for ΛCDM.
    t(z) = (1/H0) ∫_z^∞ dz' / [(1+z') E(z')]
    """
    Omega_L = 1.0 - Omega_m - config.PLANCK_2018['Omega_r']
    H0_si = H0 * config.km_s_Mpc_to_1_per_s

    # Integrate from z to large z
    z_max = 2000.0
    z_grid = np.linspace(z, z_max, 50000)
    dz = z_grid[1] - z_grid[0]

    E = np.sqrt(Omega_m * (1 + z_grid)**3
                + config.PLANCK_2018['Omega_r'] * (1 + z_grid)**4
                + Omega_L)

    from scipy.integrate import trapezoid
    integral = trapezoid(1.0 / ((1 + z_grid) * E), z_grid)
    return integral / H0_si


def model_vs_lcdm_chi2(model, sne_data):
    """
    Compute the χ² for both the merger model and ΛCDM baseline
    against the same data, for direct comparison.

    Parameters
    ----------
    model : MergerModel (solved)
    sne_data : dict

    Returns
    -------
    chi2_model, chi2_lcdm : float
    """
    # Model χ²
    z_sne = sne_data['z']
    mu_model = np.array([model.universe.distance_modulus(z) for z in z_sne])
    chi2_model = chi2_sne(mu_model, sne_data['mu'], sne_data['mu_err'])

    # ΛCDM χ² (use stored baseline if available)
    if model.baseline_results is not None:
        mu_lcdm = np.interp(z_sne, model.baseline_results['z'],
                            model.baseline_results['mu'])
    else:
        # Compute directly
        lcdm_model = MergerModel({'Omega_m_univ': config.PLANCK_2018['Omega_m'],
                                 'H0_univ': config.PLANCK_2018['H0']})
        lcdm_model.universe.solve(rho_DE_func=None)
        mu_lcdm = np.array([lcdm_model.universe.distance_modulus(z) for z in z_sne])

    chi2_lcdm = chi2_sne(mu_lcdm, sne_data['mu'], sne_data['mu_err'])

    return chi2_model, chi2_lcdm


def bic(chi2, n_params, n_data):
    """
    Bayesian Information Criterion.
    BIC = χ² + k ln(n)
    Lower BIC → better model (penalizes complexity).

    Parameters
    ----------
    chi2 : float
        Best-fit χ²
    n_params : int
        Number of free parameters
    n_data : int
        Number of data points

    Returns
    -------
    bic_value : float
    """
    return chi2 + n_params * np.log(n_data)


def aic(chi2, n_params):
    """
    Akaike Information Criterion.
    AIC = χ² + 2k
    """
    return chi2 + 2 * n_params
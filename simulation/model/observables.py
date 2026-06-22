"""
Observables computation module.
Computes cosmological observables from either the MergerModel or ΛCDM:
- Hubble parameter H(z)
- Distance modulus μ(z) for SN Ia
- Angular diameter distance D_A(z)
- Growth factor D(z) and growth rate f(z)
- Comoving volume elements
- Galaxy number counts (for JWST high-z comparisons)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from . import config


def hubble_parameter(Omega_m, Omega_r, Omega_Lambda, H0=67.4, z=None, a=None):
    """
    ΛCDM Hubble parameter: H(z) = H0 * sqrt(Ω_m(1+z)^3 + Ω_r(1+z)^4 + Ω_Λ).

    Parameters
    ----------
    Omega_m, Omega_r, Omega_Lambda : float
        Density parameters today
    H0 : float
        Hubble constant today (km/s/Mpc)
    z : array-like or float
        Redshift(s)
    a : array-like or float
        Scale factor(s) (alternative to z)

    Returns
    -------
    H : array-like
        Hubble parameter in km/s/Mpc
    """
    if a is not None:
        z = config.a_to_z(a)
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_Lambda)


def comoving_distance(z, H_func, z_max=20.0, n_steps=10000):
    """
    Comoving distance: D_C(z) = c ∫_0^z dz'/H(z').

    Parameters
    ----------
    z : array-like
        Redshift values
    H_func : callable
        Function H(z) that returns Hubble parameter in km/s/Mpc
    z_max : float
        Maximum redshift for integration grid
    n_steps : int
        Number of integration steps

    Returns
    -------
    D_C : array-like
        Comoving distance in Mpc
    """
    z_grid = np.linspace(0, max(np.max(z), z_max), n_steps)
    H_grid = H_func(z_grid)
    H_grid_si = H_grid * config.km_s_Mpc_to_1_per_s

    # Integrate dz/H(z)
    integrand = 1.0 / H_grid_si  # units: seconds
    D_C_grid = config.c * cumulative_trapz(integrand, z_grid) / config.Mpc_to_m

    return np.interp(z, z_grid, D_C_grid)


def cumulative_trapz(y, x):
    """Cumulative trapezoidal integration."""
    return np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def distance_modulus_mu(z, H_func):
    """
    Distance modulus: μ(z) = 5 log10(D_L / 10 pc).

    Parameters
    ----------
    z : array-like
        Redshifts
    H_func : callable
        H(z) in km/s/Mpc

    Returns
    -------
    mu : array-like
        Distance moduli
    """
    D_C = comoving_distance(z, H_func)
    D_L = (1.0 + z) * D_C  # Luminosity distance (flat universe)
    return 5.0 * np.log10(D_L * 1e5)  # D_L in Mpc, convert to pc


def growth_factor_lcdm(z, Omega_m):
    """
    Linear growth factor for ΛCDM (approximate formula).
    D(z) = H(z) ∫_z^∞ (1+z') / H(z')^3 dz'  normalized.

    Parameters
    ----------
    z : array-like
        Redshifts
    Omega_m : float
        Matter density today

    Returns
    -------
    D : array-like
        Growth factor (D(0)=1)
    """
    H0 = config.PLANCK_2018['H0']
    Omega_L = 1.0 - Omega_m - config.PLANCK_2018['Omega_r']

    def H(z):
        return hubble_parameter(Omega_m, config.PLANCK_2018['Omega_r'], Omega_L, H0, z=z)

    z_grid = np.linspace(0, max(np.max(z), 10.0), 5000)
    a_grid = config.z_to_a(z_grid)
    H_grid = H(z_grid) * config.km_s_Mpc_to_1_per_s

    # D(a) ∝ H(a) ∫_0^a da' / (a'^3 H(a')^3)
    integrand = 1.0 / (a_grid**3 * H_grid**3)
    integral = cumulative_trapz(integrand, a_grid)
    D = H_grid * integral
    D /= D[-1]

    return np.interp(z, z_grid[::-1], D[::-1])


def growth_rate_lcdm(z, Omega_m):
    """
    Growth rate f(z) = d ln D / d ln a for ΛCDM.
    Approximation: f(z) ≈ Ω_m(z)^γ with γ ≈ 0.55.
    """
    z_arr = np.atleast_1d(z)
    D = growth_factor_lcdm(z_arr, Omega_m)
    a = config.z_to_a(z_arr)

    dD_da = np.gradient(D, a)
    f = a / D * dD_da

    if np.ndim(z) == 0:
        return float(f[0])
    return f


def growth_rate_approximation(z, Omega_m):
    """
    Approximation: f(z) ≈ Ω_m(z)^0.55 (Linder 2005).
    """
    Omega_m_z = Omega_m * (1 + z)**3 / (Omega_m * (1 + z)**3 + 1 - Omega_m)
    return Omega_m_z**0.55


def sigma8_z(z, Omega_m, sigma8_0=0.811):
    """
    σ_8(z) = σ_8(0) * D(z) / D(0).

    Parameters
    ----------
    z : array-like
        Redshifts
    Omega_m : float
        Matter density today
    sigma8_0 : float
        σ_8 today (Planck 2018: 0.811)

    Returns
    -------
    sigma8_z : array-like
        σ_8 at redshift z
    """
    D = growth_factor_lcdm(z, Omega_m)
    return sigma8_0 * D


def hubble_tension_significance(H0_model, H0_local=73.04, sigma_local=1.04,
                                 H0_cmb=67.4, sigma_cmb=0.5):
    """
    Compute the significance of Hubble tension resolution.

    Parameters
    ----------
    H0_model : float
        Model-predicted H0
    H0_local : float
        Local measurement H0 (Riess 2022)
    sigma_local : float
        Uncertainty on local H0
    H0_cmb : float
        CMB-inferred H0 (Planck 2018)
    sigma_cmb : float
        Uncertainty on CMB H0

    Returns
    -------
    dict with chi2 and p-value metrics
    """
    chi2_local = (H0_model - H0_local)**2 / sigma_local**2
    chi2_cmb = (H0_model - H0_cmb)**2 / sigma_cmb**2
    chi2_total = chi2_local + chi2_cmb

    return {
        'chi2_local': chi2_local,
        'chi2_cmb': chi2_cmb,
        'chi2_total': chi2_total,
        'delta_local': H0_model - H0_local,
        'delta_cmb': H0_model - H0_cmb,
    }
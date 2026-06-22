"""
Interface overlap and scalar field dark energy generation.
Models dark energy as emerging from the overlap of universe branes.

The rho_DE(a) function is parameterized to mimic the behavior expected from
two expanding branes that start far apart, eventually overlap, generating
a repulsive scalar field. The key physical features:
- At early times (small a): no overlap, no extra DE → rho_DE ≈ 0
- At a = a_contact: branes first touch, DE begins
- At a > a_contact: overlap grows, DE density rises
- DE growth is nonlinear (V_overlap^beta)

For full coupled Friedmann + brane dynamics, one would need to solve
both systems simultaneously. Here we provide a parameterization that
captures the essential physics for comparison with data.
"""

import numpy as np
from . import config


def rho_DE_interface(a, alpha=1.0, a_contact=0.5, beta=2.0, smoothness=10.0):
    """
    Dark energy density from universe interface overlap.

    Physical model: Two universes with comoving radii r = a expand in a bulk.
    They start with separation D. At a = a_contact, their boundaries touch.
    For a > a_contact, overlap begins.

    The overlap volume V_overlap ∝ (a - a_overlap)^beta above threshold.
    Dark energy density rho_DE ∝ V_overlap^beta.

    Parameters
    ----------
    a : array-like
        Scale factor
    alpha : float
        Amplitude of DE (calibrated to give Omega_DE ~ 0.7 at a=1)
    a_contact : float
        Scale factor at first contact (i.e., when DE starts turning on)
    beta : float
        Power-law growth of DE after contact
    smoothness : float
        Smoothness of the turn-on transition

    Returns
    -------
    rho_DE : array-like
        Dark energy density in units of LCDM's cosmological constant density
        So rho_DE = 1.0 corresponds to Omega_Lambda = 0.685
    """
    a = np.asarray(a, dtype=float)

    # Smooth step function for the onset of DE
    # sigmoid centered at a_contact
    activation = 1.0 / (1.0 + np.exp(-smoothness * (a - a_contact)))

    # For a < a_contact: rho_DE ≈ 0 (exponentially small)
    # For a > a_contact: rho_DE grows as (a - a_contact)^beta, then saturates
    a_diff = np.maximum(a - a_contact, 0.0)

    # Saturation: overlap can't exceed the full volume of a universe
    # rho_DE saturates when a_diff >> a_contact
    saturation = 1.0 - np.exp(-a_diff)

    # DE density
    rho_DE = alpha * saturation**beta * activation

    return rho_DE


def get_rho_DE_func(alpha=1.0, a_contact=0.5, beta=2.0, smoothness=10.0):
    """
    Returns rho_DE(a) in kg/m^3.

    Calibration: at z=0 (a=1), rho_DE(1) should equal
    Omega_Lambda * rho_crit0 for flat ΛCDM-equivalent behavior.

    The interface function gives a dimensionless fraction. We multiply by
    the target density and divide by the interface function value at a=1
    so that the output is CORRECTLY CALIBRATED.
    """
    # Use the Universe's actual H0 and Omega parameters to compute rho_crit0
    Omega_m = config.MERGER_DEFAULTS['Omega_m_univ']
    Omega_r = config.MERGER_DEFAULTS['Omega_r_univ']
    Omega_L = 1.0 - Omega_m - Omega_r
    H0 = config.MERGER_DEFAULTS['H0_univ'] * config.km_s_Mpc_to_1_per_s
    rho_crit0 = 3.0 * H0**2 / (8.0 * np.pi * config.G)
    rho_target = Omega_L * rho_crit0

    # Compute interface function at a=1 for normalization
    frac_at_1 = rho_DE_interface(1.0, alpha, a_contact, beta, smoothness)
    if abs(frac_at_1) < 1e-30:
        frac_at_1 = 1e-30

    # Normalization factor: ensures rho_DE(1) = rho_target
    norm = rho_target / frac_at_1

    def rho_DE_func(a):
        frac = rho_DE_interface(a, alpha, a_contact, beta, smoothness)
        return frac * norm

    return rho_DE_func

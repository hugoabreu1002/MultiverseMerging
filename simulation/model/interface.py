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

    # Compute interface function at a=1 for normalization using alpha=1
    # so that `alpha` acts as a relative amplitude (not part of the
    # normalization denominator). This avoids dividing by a value
    # that could be numerically ~0 when the interface is nearly off at a=1.
    frac_at_1 = rho_DE_interface(1.0, alpha=1.0, a_contact=a_contact,
                                 beta=beta, smoothness=smoothness)
    weighted = alpha * frac_at_1
    # Safety floor to avoid pathologically large normalization
    if abs(weighted) < 1e-12:
        weighted = 1e-12

    # Normalization factor: ensures rho_DE(1) = Omega_L * rho_crit0
    norm = rho_target / weighted

    def rho_DE_func(a):
        frac = rho_DE_interface(a, alpha=1.0, a_contact=a_contact,
                                 beta=beta, smoothness=smoothness)
        return alpha * frac * norm

    return rho_DE_func


def get_rho_DE_multi(N_universes=2, alpha=1.0, a_contact=0.5, beta=2.0,
                     smoothness=10.0, a_contact_spread=0.2, alpha_list=None,
                     beta_list=None):
    """
    Construct a rho_DE(a) that is the sum of contributions from multiple
    external universes. By default contributions are staggered in contact
    time across a small range and share the same beta; amplitudes share
    the provided `alpha` (split evenly) unless `alpha_list` is given.
    """
    # Compute target density (same calibration as get_rho_DE_func)
    Omega_m = config.MERGER_DEFAULTS['Omega_m_univ']
    Omega_r = config.MERGER_DEFAULTS['Omega_r_univ']
    Omega_L = 1.0 - Omega_m - Omega_r
    H0 = config.MERGER_DEFAULTS['H0_univ'] * config.km_s_Mpc_to_1_per_s
    rho_crit0 = 3.0 * H0**2 / (8.0 * np.pi * config.G)
    rho_target = Omega_L * rho_crit0

    # Build per-universe parameter lists
    if N_universes <= 1:
        return get_rho_DE_func(alpha=alpha, a_contact=a_contact,
                               beta=beta, smoothness=smoothness)

    if alpha_list is None:
        alpha_list = [alpha / float(N_universes)] * N_universes
    if beta_list is None:
        beta_list = [beta] * N_universes

    # Stagger contact times between a_contact and min(1.0, a_contact+a_contact_spread)
    a_max = min(1.0, a_contact + abs(a_contact_spread))
    a_contacts = np.linspace(a_contact, a_max, N_universes)

    # Compute per-universe fraction at a=1 (with alpha=1) and build weighted sum
    fracs_at_1 = [rho_DE_interface(1.0, alpha=1.0, a_contact=ac,
                                   beta=beta_list[i], smoothness=smoothness)
                  for i, ac in enumerate(a_contacts)]

    weighted_total = sum([alpha_list[i] * fracs_at_1[i] for i in range(N_universes)])
    if abs(weighted_total) < 1e-12:
        weighted_total = 1e-12
    norm = rho_target / weighted_total

    def rho_DE_func(a):
        total = 0.0
        for i in range(N_universes):
            frac = rho_DE_interface(a, alpha=1.0, a_contact=a_contacts[i],
                                     beta=beta_list[i], smoothness=smoothness)
            total += alpha_list[i] * frac
        return norm * total

    return rho_DE_func

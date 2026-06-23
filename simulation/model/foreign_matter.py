"""
Foreign dark matter from merging universes.
Models the gravitational influence of actual physical matter from other
universes that interacts with our universe during merging events.

Two geometries are modeled:
- 5D Spatial: Universes are branes separated along a 5th spatial axis.
  Foreign matter crosses the gap via spatial overlap, same geometry as DE.
- 4D Temporal: Universes are quantum histories across time. Foreign matter
  leaks via quantum tunneling between adjacent time-slice universes.
"""

import numpy as np
from . import config


def rho_DM_5D_spatial(a, alpha_dm=0.15, a_contact=0.3, beta_dm=1.5,
                      smoothness=10.0, overdensity_factor=1.0):
    """
    Dark matter density from foreign matter crossing via 5D spatial overlap.
    
    Same overlap geometry as the DE interface model: two universes expanding
    in a bulk, touching at a_contact, with overlap volume growing afterwards.
    The foreign matter is actual physical matter (stars, gas, dark matter halos)
    from the other universe that becomes gravitationally coupled to ours.
    
    Parameters
    ----------
    a : array-like
        Scale factor
    alpha_dm : float
        Amplitude of foreign DM (calibrated to give Omega_DM ~ 0.27 at z=0)
    a_contact : float
        Scale factor at first contact (when foreign matter starts appearing)
    beta_dm : float
        Power-law growth of foreign matter influx after contact
    smoothness : float
        Smoothness of the turn-on transition
    overdensity_factor : float
        Enhancement in high-density regions due to gravitational focusing
    
    Returns
    -------
    rho_DM : array-like
        Foreign dark matter density in kg/m^3
    """
    a = np.asarray(a, dtype=float)
    
    # Smooth step function for the onset of foreign matter
    activation = 1.0 / (1.0 + np.exp(-smoothness * (a - a_contact)))
    
    # For a < a_contact: rho_DM ≈ 0 (universes are separate)
    # For a > a_contact: foreign matter influx grows
    a_diff = np.maximum(a - a_contact, 0.0)
    
    # Saturation: overlap can't exceed the full volume
    saturation = 1.0 - np.exp(-a_diff)
    
    # Foreign matter density
    rho_DM = alpha_dm * saturation**beta_dm * activation * overdensity_factor
    
    return rho_DM


def rho_DM_4D_temporal(a, alpha_dm=0.15, a_contact=0.3, beta_dm=1.5,
                       smoothness=10.0, tunneling_scale=0.15,
                       time_dilation_enhancement=0.0):
    """
    Dark matter density from foreign matter leaking via quantum tunneling
    across time-slice universes (4D temporal multiverse).
    
    In this model, the multiverse exists across quantum time. "Merging"
    is quantum interference between different time slices. Foreign matter
    tunnels between these slices with probability P ∝ exp(-Δt²/σ²).
    
    The turn-on is more gradual than the 5D model because time-slice
    overlap is inherently quantum and non-local.
    
    Parameters
    ----------
    a : array-like
        Scale factor
    alpha_dm : float
        Amplitude of foreign DM
    a_contact : float
        Scale factor where quantum overlap becomes significant
    beta_dm : float
        Power-law growth of tunneling influx
    smoothness : float
        Smoothness of the transition
    tunneling_scale : float
        Width of the quantum tunneling Gaussian (smaller = sharper turn-on)
    time_dilation_enhancement : float
        Enhancement from gravitational time dilation (0 = none, 1 = full)
        Represents: deeper potential wells → more time dilation → more tunneling
    
    Returns
    -------
    rho_DM : array-like
        Foreign dark matter density in kg/m^3
    """
    a = np.asarray(a, dtype=float)
    
    # Quantum tunneling probability between time slices
    # Gaussian overlap in time: exp(-(a - a_contact)^2 / (2 * tunneling_scale^2))
    # This gives a softer, more gradual turn-on than the sigmoid
    a_diff = a - a_contact
    
    # Quantum overlap: Gaussian activation
    quantum_overlap = np.exp(-a_diff**2 / (2.0 * tunneling_scale**2))
    
    # For a < a_contact: very small quantum overlap (tail of Gaussian)
    # For a > a_contact: overlap grows, then saturates
    # Use cumulative distribution of Gaussian (error function) for the
    # integrated tunneling probability
    from scipy.special import erfc
    # erfc(-x) / 2 gives the cumulative of a Gaussian from -inf to x
    # This ensures: at a << a_contact, P ≈ 0; at a >> a_contact, P ≈ 1
    tunneling_prob = 0.5 * erfc(-a_diff / (np.sqrt(2.0) * tunneling_scale))
    
    # Power-law modulation for non-linear growth
    power_law = np.maximum(a_diff, 0.0)**beta_dm
    
    # Time dilation enhancement: regions with stronger gravity
    # have more dilated time, which increases the effective tunneling rate
    # This is a local effect, parameterized here as a global average
    td_boost = 1.0 + time_dilation_enhancement * tunneling_prob
    
    # Foreign matter density from temporal quantum tunneling
    # The combination gives: gradual turn-on (erfc) + power-law growth
    # + time dilation enhancement
    rho_DM = alpha_dm * tunneling_prob * (1.0 + power_law) * td_boost
    
    return rho_DM


def rho_DE_4D_temporal(a, alpha_de=0.3, a_contact=0.3, beta_de=2.0,
                       smoothness=10.0, interference_scale=0.2):
    """
    Dark energy density from quantum interference across time-slice universes.
    
    In the 4D temporal model, dark energy is not from spatial brane overlap
    but from constructive interference of vacuum energy across adjacent
    quantum time slices. When time slices are far apart (a << a_contact),
    interference is negligible. When they begin to overlap quantum-mechanically,
    constructive interference generates a repulsive vacuum energy.
    
    Parameters
    ----------
    a : array-like
        Scale factor
    alpha_de : float
        Amplitude of temporal DE
    a_contact : float
        Scale factor where quantum overlap begins
    beta_de : float
        Power-law growth of interference energy
    smoothness : float
        Smoothness of transition
    interference_scale : float
        Scale of quantum interference (larger = longer-range interference)
    
    Returns
    -------
    rho_DE : array-like
        Dark energy density in kg/m^3
    """
    a = np.asarray(a, dtype=float)
    
    # In the 4D temporal model, the onset is smoother because
    # quantum overlap has a non-zero tail even at a < a_contact
    a_diff = a - a_contact
    
    # Quantum interference envelope: Voigt-like profile
    # Gaussian core + Lorentzian tails for long-range quantum effects
    gauss = np.exp(-a_diff**2 / (2.0 * interference_scale**2))
    lorentz = 1.0 / (1.0 + (a_diff / interference_scale)**2)
    
    # Mix: dominant Gaussian at early times, Lorentzian at late times
    interference_profile = 0.7 * gauss + 0.3 * lorentz
    
    # For a < a_contact: small but non-zero (quantum tail)
    # For a > a_contact: grows, then saturates via sigmoid
    # The interference energy plateaus at late times
    from scipy.special import erfc
    temporal_activation = 0.5 * erfc(-a_diff / (np.sqrt(2.0) * interference_scale))
    
    # Power-law growth
    a_diff_pos = np.maximum(a_diff, 0.0)
    growth = a_diff_pos**beta_de / (1.0 + a_diff_pos**beta_de)  # saturation
    
    # DE density from temporal quantum interference
    rho_DE = alpha_de * temporal_activation * growth * interference_profile
    
    return rho_DE


def get_rho_DM_func(geometry='5D', alpha_dm=0.15, a_contact=0.3,
                    beta_dm=1.5, smoothness=10.0, overdensity_factor=1.0,
                    tunneling_scale=0.15, time_dilation_enhancement=0.0):
    """
    Returns rho_DM(a) in kg/m^3 for the chosen geometry.
    
    Calibration: at z=0 (a=1), rho_DM(1) should equal Omega_DM * rho_crit0
    where Omega_DM ≈ 0.264 (the non-baryonic dark matter fraction).
    """
    Omega_m = config.MERGER_DEFAULTS['Omega_m_univ']
    Omega_b = config.PLANCK_2018['Omega_b']  # Baryon density
    Omega_DM_target = Omega_m - Omega_b  # Dark matter fraction ~0.2657
    
    Omega_r = config.MERGER_DEFAULTS['Omega_r_univ']
    H0 = config.MERGER_DEFAULTS['H0_univ'] * config.km_s_Mpc_to_1_per_s
    rho_crit0 = 3.0 * H0**2 / (8.0 * np.pi * config.G)
    rho_target = Omega_DM_target * rho_crit0
    
    # Compute normalization factor
    if geometry == '5D':
        frac_at_1 = rho_DM_5D_spatial(1.0, alpha_dm=1.0, a_contact=a_contact,
                                       beta_dm=beta_dm, smoothness=smoothness,
                                       overdensity_factor=overdensity_factor)
    else:  # 4D
        frac_at_1 = rho_DM_4D_temporal(1.0, alpha_dm=1.0, a_contact=a_contact,
                                        beta_dm=beta_dm, smoothness=smoothness,
                                        tunneling_scale=tunneling_scale,
                                        time_dilation_enhancement=time_dilation_enhancement)
    
    weighted = alpha_dm * frac_at_1
    if abs(weighted) < 1e-12:
        weighted = 1e-12
    
    norm = rho_target / weighted
    
    def rho_DM_func(a):
        if geometry == '5D':
            frac = rho_DM_5D_spatial(a, alpha_dm=1.0, a_contact=a_contact,
                                      beta_dm=beta_dm, smoothness=smoothness,
                                      overdensity_factor=overdensity_factor)
        else:  # 4D
            frac = rho_DM_4D_temporal(a, alpha_dm=1.0, a_contact=a_contact,
                                       beta_dm=beta_dm, smoothness=smoothness,
                                       tunneling_scale=tunneling_scale,
                                       time_dilation_enhancement=time_dilation_enhancement)
        return alpha_dm * frac * norm
    
    return rho_DM_func


def get_rho_DE_temporal_func(alpha_de=0.3, a_contact=0.3, beta_de=2.0,
                              smoothness=10.0, interference_scale=0.2):
    """
    Returns rho_DE(a) in kg/m^3 for the 4D temporal model.
    Calibrated to give Omega_DE ~ 0.685 at z=0.
    """
    Omega_m = config.MERGER_DEFAULTS['Omega_m_univ']
    Omega_r = config.MERGER_DEFAULTS['Omega_r_univ']
    Omega_L_target = 1.0 - Omega_m - Omega_r
    
    H0 = config.MERGER_DEFAULTS['H0_univ'] * config.km_s_Mpc_to_1_per_s
    rho_crit0 = 3.0 * H0**2 / (8.0 * np.pi * config.G)
    rho_target = Omega_L_target * rho_crit0
    
    # Compute normalization with alpha=1
    frac_at_1 = rho_DE_4D_temporal(1.0, alpha_de=1.0, a_contact=a_contact,
                                    beta_de=beta_de, smoothness=smoothness,
                                    interference_scale=interference_scale)
    
    weighted = alpha_de * frac_at_1
    if abs(weighted) < 1e-12:
        weighted = 1e-12
    
    norm = rho_target / weighted
    
    def rho_DE_func(a):
        frac = rho_DE_4D_temporal(a, alpha_de=1.0, a_contact=a_contact,
                                   beta_de=beta_de, smoothness=smoothness,
                                   interference_scale=interference_scale)
        return alpha_de * frac * norm
    
    return rho_DE_func


def foreign_dm_fraction(z, rho_DM_func, rho_total_func):
    """
    Compute the fraction of total density that is foreign DM.
    Ω_DM_foreign(z) = ρ_DM_foreign(z) / ρ_total(z)
    
    Parameters
    ----------
    z : array-like
        Redshifts
    rho_DM_func : callable
        Function rho_DM(a) returning foreign DM density
    rho_total_func : callable
        Function rho_total(a) returning total density
    
    Returns
    -------
    f_DM : array-like
        Foreign DM fraction at each redshift
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    
    rho_DM = np.array([rho_DM_func(ai) for ai in a])
    rho_total = np.array([rho_total_func(ai) for ai in a])
    
    return rho_DM / (rho_total + 1e-30)


def overdensity_evolution(z, Omega_m=0.315):
    """
    Linear overdensity growth factor δ(z) ∝ D(z) for matter.
    Used to estimate where foreign DM would cluster most.
    
    Returns the relative enhancement factor for high-density regions.
    """
    # Approximation: δ(z) ∝ 1/(1+z) at early times, suppressed at late times
    # For ΛCDM-like growth
    from .observables import growth_factor_lcdm
    D = growth_factor_lcdm(z, Omega_m)
    return D
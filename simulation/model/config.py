"""
Physical constants and configuration for the merging universes model.
All values in SI units unless otherwise noted.
"""

# === Fundamental Constants ===
G = 6.67430e-11          # gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8         # speed of light (m/s)
H0_Planck = 67.4e3 / (3.0857e22)  # H0 = 67.4 km/s/Mpc → 1/s (Planck 2018)
H0_Riess = 73.04e3 / (3.0857e22)  # H0 = 73.04 km/s/Mpc → 1/s (Riess 2022)

# Convert: 1 Mpc = 3.0857e22 m
Mpc_to_m = 3.085677581e22
km_s_Mpc_to_1_per_s = 1e3 / Mpc_to_m

# === Current Best-Fit ΛCDM Parameters (Planck 2018 TT+lowE+lensing) ===
PLANCK_2018 = {
    'H0': 67.4,            # km/s/Mpc
    'Omega_m': 0.315,      # matter density parameter today
    'Omega_b': 0.0493,     # baryon density
    'Omega_r': 9.2e-5,     # radiation density (CMB + neutrinos)
    'Omega_Lambda': 0.685, # dark energy density (ΛCDM)
    'Omega_k': 0.0,        # curvature
    'sigma8': 0.811,       # clustering amplitude
    'ns': 0.965,           # spectral index
}

# === Merging Universe Model Parameters (defaults) ===
MERGER_DEFAULTS = {
    # Interface DE parameters (rho_DE parameterization)
    'alpha': 0.3,          # DE coupling strength (amplitude, calibrated for Omega_DE ~ 0.7)
    'beta': 2.5,           # nonlinear growth exponent after contact
    'a_contact': 0.3,      # scale factor at first interface contact (higher = earlier DE)

    # Legacy brane parameters (not used in simplified parameterization)
    'd_init': 3.0,
    'v_init': 0.3,
    'N_universes': 2,      # number of universes

    # Each universe's composition (can differ)
    'Omega_m_univ': 0.315,
    'Omega_r_univ': 9.2e-5,
    'Omega_k_univ': 0.0,
    'H0_univ': 67.4,       # km/s/Mpc

    # Numerical
    'z_max': 20.0,          # maximum redshift for integration
    'n_steps': 10000,       # number of time steps
    'tol': 1e-8,            # ODE solver tolerance
}

# === Redshift to scale factor conversion ===
def z_to_a(z):
    """Convert redshift z to scale factor a = 1/(1+z)."""
    return 1.0 / (1.0 + z)

def a_to_z(a):
    """Convert scale factor a to redshift z = 1/a - 1."""
    return 1.0 / a - 1.0

# === Standard ruler: sound horizon at drag epoch (Mpc) ===
R_DRAG_LCDM = 147.09  # Mpc, from Planck 2018

# === Unit conversions ===
SECONDS_PER_GYR = 3.15576e16    # seconds per gigayear
GYR_PER_SECOND = 1.0 / SECONDS_PER_GYR
"""
Single universe Friedmann dynamics.
Uses direct integration on a uniform redshift grid.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from . import config


class Universe:
    """Represents a single universe with its own expansion history."""

    def __init__(self, params=None, label='Universe'):
        self.label = label
        self.params = config.MERGER_DEFAULTS.copy()
        if params:
            self.params.update(params)
        H0_km = self.params.get('H0_univ', self.params.get('H0', 67.4))
        self.H0 = H0_km * config.km_s_Mpc_to_1_per_s
        self.H0_km = H0_km
        self.Omega_m = self.params['Omega_m_univ']
        self.Omega_r = self.params['Omega_r_univ']
        self.Omega_k = self.params['Omega_k_univ']
        self.Omega_Lambda = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k
        self.rho_crit0 = 3.0 * self.H0**2 / (8.0 * np.pi * config.G)
        self.z_arr = None
        self.a_arr = None
        self.t_arr = None
        self.H_arr = None
        self.rho_m_arr = None
        self.rho_r_arr = None
        self.rho_DE_arr = None

    def solve(self, rho_DE_func=None):
        n_steps = self.params['n_steps']
        z_max = self.params['z_max']
        self.z_arr = np.linspace(0, z_max, n_steps)
        self.a_arr = 1.0 / (1.0 + self.z_arr)

        rho_m_arr = self.rho_crit0 * self.Omega_m * (1 + self.z_arr)**3
        rho_r_arr = self.rho_crit0 * self.Omega_r * (1 + self.z_arr)**4

        if rho_DE_func is not None:
            self.rho_DE_arr = np.array([rho_DE_func(a) for a in self.a_arr])
            self.rho_DE_arr = np.maximum(self.rho_DE_arr, 0.0)
        else:
            self.rho_DE_arr = self.rho_crit0 * self.Omega_Lambda * np.ones_like(self.z_arr)

        rho_total = rho_m_arr + rho_r_arr + self.rho_DE_arr
        H_raw = np.sqrt(8.0 * np.pi * config.G * rho_total / 3.0)

        # Normalize so that H(z=0) = self.H0
        # Only scale rho_DE; matter/radiation already use self.rho_crit0
        H0_raw = H_raw[0]
        if H0_raw > 0:
            scale = (self.H0 / H0_raw)**2
            self.rho_DE_arr *= scale
            rho_total = rho_m_arr + rho_r_arr + self.rho_DE_arr
            H_raw = np.sqrt(8.0 * np.pi * config.G * rho_total / 3.0)

        self.H_arr = H_raw

        # Cosmic time: t(z) = ∫_z^∞ dz' / [(1+z') H(z')]
        integrand = 1.0 / ((1.0 + self.z_arr) * self.H_arr)
        t_from_zmax = cumulative_trapezoid(integrand[::-1], self.z_arr[::-1], initial=0)
        self.t_arr = t_from_zmax[::-1]

        self.rho_m_arr = rho_m_arr
        self.rho_r_arr = rho_r_arr
        return {
            'z': self.z_arr, 'a': self.a_arr, 't': self.t_arr,
            'H': self.H_arr, 'rho_m': rho_m_arr, 'rho_r': rho_r_arr,
            'rho_DE': self.rho_DE_arr,
        }

    def H_of_z(self, z):
        if self.z_arr is None:
            raise RuntimeError("Must call solve() first.")
        return np.interp(z, self.z_arr, self.H_arr)

    def t_of_z(self, z):
        if self.t_arr is None:
            raise RuntimeError("Must call solve() first.")
        return np.interp(z, self.z_arr, self.t_arr)

    def dc_of_z(self, z):
        """Comoving distance D_C(z) = c ∫_0^z dz'/H(z') in Mpc."""
        if self.z_arr is None:
            raise RuntimeError("Must call solve() first.")
        integral = cumulative_trapezoid(1.0 / self.H_arr, self.z_arr, initial=0)
        dc = config.c * integral / config.Mpc_to_m
        if np.ndim(z) == 0:
            return float(np.interp(z, self.z_arr, dc))
        return np.interp(z, self.z_arr, dc)

    def luminosity_distance(self, z):
        dc = self.dc_of_z(z)
        if np.ndim(z) == 0:
            return (1.0 + z) * dc
        return (1.0 + z) * dc

    def angular_diameter_distance(self, z):
        dc = self.dc_of_z(z)
        if np.ndim(z) == 0:
            return dc / (1.0 + z)
        return dc / (1.0 + z)

    def distance_modulus(self, z):
        dc = self.dc_of_z(z)
        DL = (1.0 + z) * dc
        DL_safe = np.maximum(DL, 1e-10)
        return 5.0 * np.log10(DL_safe) + 25.0

    def growth_factor(self, z):
        if self.z_arr is None:
            raise RuntimeError("Must call solve() first.")
        a = self.a_arr
        H = self.H_arr
        mask = a > 1e-4
        a_use = a[mask]
        H_use = H[mask]
        z_use = self.z_arr[mask]
        integrand = 1.0 / (a_use**3 * H_use**3)
        integral = cumulative_trapezoid(integrand, a_use, initial=0)
        D = H_use * integral
        # Regularize to avoid zeros/division issues
        eps = 1e-12
        D = np.maximum(D, eps)
        if D[-1] <= 0:
            D[-1] = eps
        D /= D[-1]
        return np.interp(z, z_use[::-1], D[::-1])

    def growth_rate(self, z):
        if self.z_arr is None:
            raise RuntimeError("Must call solve() first.")
        mask = self.a_arr > 1e-4
        a = self.a_arr[mask]
        D = self.growth_factor(self.z_arr[mask])
        # Ensure D is finite and positive
        eps = 1e-12
        D = np.maximum(D, eps)
        dD_da = np.gradient(D, a)
        f_raw = a / D * dD_da
        # Replace NaNs/infs with zero
        f_raw = np.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0)
        f = np.where(D > eps, f_raw, 0.0)
        z_use = self.z_arr[mask]
        return np.interp(z, z_use[::-1], f[::-1])
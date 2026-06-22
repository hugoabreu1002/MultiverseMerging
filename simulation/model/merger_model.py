"""
Multi-universe merger model.
Ties together the Friedmann dynamics (universe.py) with the
interface dark energy generation (interface.py) to provide
a full cosmological model.
"""

import numpy as np
from . import config
from .universe import Universe
from .interface import get_rho_DE_func


class MergerModel:
    """
    Merging universes cosmological model.
    Uses a parameterized rho_DE(a) from interface overlap.
    """

    def __init__(self, params=None):
        self.params = config.MERGER_DEFAULTS.copy()
        if params:
            self.params.update(params)

        # Extract the key DE parameters
        self.alpha = self.params.get('alpha', 1.0)
        self.beta = self.params.get('beta', 2.0)
        self.a_contact = self.params.get('a_contact', 0.5)
        self.N_universes = self.params.get('N_universes', 1)

        self.label = (f"merger_N{self.params.get('N_universes',2)}_"
                      f"a{self.alpha:.2e}_b{self.beta:.1f}_"
                      f"ac{self.a_contact:.2f}")

        self.universe = Universe(self.params, label='Observed Universe')
        self.results = None
        self.baseline_results = None

    def solve(self):
        """Solve Friedmann equation with interface-generated DE."""
        # Choose single- or multi-universe rho_DE function
        if self.params.get('N_universes', 1) <= 1:
            rho_DE_func = get_rho_DE_func(
                alpha=self.alpha,
                a_contact=self.a_contact,
                beta=self.beta
            )
        else:
            rho_DE_func = get_rho_DE_func if False else None
            # Use multi-universe builder from interface
            rho_DE_func = get_rho_DE_func  # fallback
            try:
                rho_DE_func = get_rho_DE_func  # keep reference
            except Exception:
                pass
            # Construct multi-universe function
            from .interface import get_rho_DE_multi
            alpha_list = self.params.get('alpha_list', None)
            a_contact_list = self.params.get('a_contact_list', None)
            beta_list = self.params.get('beta_list', None)
            a_contact_spread = self.params.get('a_contact_spread', 0.2)

            rho_DE_func = get_rho_DE_multi(
                N_universes=self.params.get('N_universes', 2),
                alpha=self.alpha,
                a_contact=self.a_contact,
                beta=self.beta,
                smoothness=self.params.get('smoothness', 10.0),
                a_contact_spread=a_contact_spread,
                alpha_list=alpha_list,
                beta_list=beta_list
            )
        self.universe.solve(rho_DE_func=rho_DE_func)

        z = self.universe.z_arr
        H = self.universe.H_arr
        rho_m = self.universe.rho_m_arr
        rho_r = self.universe.rho_r_arr
        rho_DE = self.universe.rho_DE_arr

        self.results = {
            'z': z, 'a': self.universe.a_arr, 't': self.universe.t_arr,
            'H': H,
            'H_km_s_Mpc': H / config.km_s_Mpc_to_1_per_s,
            'rho_m': rho_m, 'rho_r': rho_r, 'rho_DE': rho_DE,
            'Omega_m': rho_m / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
            'Omega_DE': rho_DE / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
        }

        z_vals = z
        self.results['DA'] = np.array([self.universe.angular_diameter_distance(zi) for zi in z_vals])
        self.results['DL'] = np.array([self.universe.luminosity_distance(zi) for zi in z_vals])
        self.results['mu'] = np.array([self.universe.distance_modulus(zi) for zi in z_vals])
        self.results['D'] = self.universe.growth_factor(z_vals)
        self.results['f'] = self.universe.growth_rate(z_vals)

        return self.results

    def solve_lcdm(self):
        """Solve LCDM baseline for comparison."""
        lcdm = Universe(self.params, label='LCDM')
        lcdm.solve(rho_DE_func=None)
        z = lcdm.z_arr
        H = lcdm.H_arr
        rho_m = lcdm.rho_m_arr
        rho_r = lcdm.rho_r_arr
        rho_DE = lcdm.rho_DE_arr

        self.baseline_results = {
            'z': z, 'a': lcdm.a_arr, 't': lcdm.t_arr,
            'H': H,
            'H_km_s_Mpc': H / config.km_s_Mpc_to_1_per_s,
            'rho_m': rho_m, 'rho_r': rho_r, 'rho_DE': rho_DE,
            'Omega_m': rho_m / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
            'Omega_DE': rho_DE / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
        }

        z_vals = z
        self.baseline_results['DA'] = np.array([lcdm.angular_diameter_distance(zi) for zi in z_vals])
        self.baseline_results['DL'] = np.array([lcdm.luminosity_distance(zi) for zi in z_vals])
        self.baseline_results['mu'] = np.array([lcdm.distance_modulus(zi) for zi in z_vals])
        self.baseline_results['D'] = lcdm.growth_factor(z_vals)
        self.baseline_results['f'] = lcdm.growth_rate(z_vals)
        return self.baseline_results

    def hubble_tension_metric(self):
        H0_model = self.params.get('H0_univ', 67.4)
        H0_planck = config.PLANCK_2018['H0']
        H0_riess = 73.04
        return {
            'H0_model': H0_model,
            'H0_Planck': H0_planck,
            'H0_Riess': H0_riess,
            'delta_vs_Planck': H0_model - H0_planck,
            'delta_vs_Riess': H0_model - H0_riess,
        }

    def summary(self):
        print("=" * 60)
        print("Merging Universes Model Summary")
        print("=" * 60)
        print("Parameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v}")
        if self.results:
            print(f"\n  H0 = {self.results['H_km_s_Mpc'][0]:.2f} km/s/Mpc")
            print(f"  Omega_m = {self.results['Omega_m'][0]:.4f}")
            print(f"  Omega_DE = {self.results['Omega_DE'][0]:.4f}")
        print("=" * 60)
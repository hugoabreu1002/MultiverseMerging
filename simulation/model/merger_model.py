"""
Multi-universe merger model.
Ties together the Friedmann dynamics (universe.py) with the
interface dark energy generation (interface.py) and foreign dark
matter (foreign_matter.py) to provide a full cosmological model
with two competing geometries: 5D spatial brane and 4D temporal
quantum multiverse.
"""

import numpy as np
from . import config
from .universe import Universe
from .interface import get_rho_DE_func, get_rho_DE_multi


class MergerModel:
    """
    Merging universes cosmological model.
    Supports two geometries:
    - '5D': Spatial brane overlap (original model)
    - '4D': Temporal quantum multiverse (time-slice interference)
    
    Both geometries can include foreign dark matter from merging universes.
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
        
        # Geometry and foreign DM settings
        self.geometry = self.params.get('geometry', '5D')
        self.include_foreign_dm = self.params.get('include_foreign_dm', False)
        self.alpha_dm = self.params.get('alpha_dm', 0.15)
        self.beta_dm = self.params.get('beta_dm', 1.5)

        self.label = (f"{self.geometry}_N{self.params.get('N_universes',2)}_"
                      f"a{self.alpha:.2e}_b{self.beta:.1f}_"
                      f"ac{self.a_contact:.2f}")
        if self.include_foreign_dm:
            self.label += f"_dm_a{self.alpha_dm:.2e}_b{self.beta_dm:.1f}"

        self.universe = Universe(self.params, label='Observed Universe')
        self.results = None
        self.baseline_results = None

    def solve(self):
        """Solve Friedmann equation with interface-generated DE and optional foreign DM."""
        
        # --- Build rho_DE(a) function according to geometry ---
        if self.geometry == '5D':
            rho_DE_func = self._build_rho_DE_5D()
        elif self.geometry == '4D':
            rho_DE_func = self._build_rho_DE_4D()
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}. Use '5D' or '4D'.")

        # --- Build rho_DM_foreign(a) function if enabled ---
        rho_DM_foreign_func = None
        if self.include_foreign_dm:
            rho_DM_foreign_func = self._build_rho_DM_func()

        # Solve Friedmann with combined DE and foreign DM
        self.universe.solve(rho_DE_func=rho_DE_func,
                            rho_DM_foreign_func=rho_DM_foreign_func)

        z = self.universe.z_arr
        H = self.universe.H_arr
        rho_m = self.universe.rho_m_arr
        rho_r = self.universe.rho_r_arr
        rho_DE = self.universe.rho_DE_arr
        rho_DM_foreign = self.universe.rho_DM_foreign_arr if self.universe.rho_DM_foreign_arr is not None else np.zeros_like(z)
        rho_m_local = self.universe.rho_m_local_arr if self.universe.rho_m_local_arr is not None else rho_m

        # Total matter: local baryonic + foreign DM
        rho_m_total = rho_m_local + rho_DM_foreign

        self.results = {
            'z': z, 'a': self.universe.a_arr, 't': self.universe.t_arr,
            'H': H,
            'H_km_s_Mpc': H / config.km_s_Mpc_to_1_per_s,
            'rho_m': rho_m_total, 'rho_r': rho_r, 'rho_DE': rho_DE,
            'rho_m_local': rho_m_local,
            'rho_DM_foreign': rho_DM_foreign,
            'Omega_m': rho_m_total / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
            'Omega_DE': rho_DE / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
            'Omega_DM_foreign': rho_DM_foreign / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
            'Omega_m_local': rho_m_local / (3.0 * H**2 / (8.0 * np.pi * config.G) + 1e-30),
        }

        z_vals = z
        self.results['DA'] = np.array([self.universe.angular_diameter_distance(zi) for zi in z_vals])
        self.results['DL'] = np.array([self.universe.luminosity_distance(zi) for zi in z_vals])
        self.results['mu'] = np.array([self.universe.distance_modulus(zi) for zi in z_vals])
        self.results['D'] = self.universe.growth_factor(z_vals)
        self.results['f'] = self.universe.growth_rate(z_vals)

        return self.results

    def _build_rho_DE_5D(self):
        """Build rho_DE(a) for 5D spatial brane model."""
        if self.params.get('N_universes', 1) <= 1:
            return get_rho_DE_func(
                alpha=self.alpha,
                a_contact=self.a_contact,
                beta=self.beta
            )
        else:
            from .interface import get_rho_DE_multi
            alpha_list = self.params.get('alpha_list', None)
            a_contact_list = self.params.get('a_contact_list', None)
            beta_list = self.params.get('beta_list', None)
            a_contact_spread = self.params.get('a_contact_spread', 0.2)

            return get_rho_DE_multi(
                N_universes=self.params.get('N_universes', 2),
                alpha=self.alpha,
                a_contact=self.a_contact,
                beta=self.beta,
                smoothness=self.params.get('smoothness', 10.0),
                a_contact_spread=a_contact_spread,
                alpha_list=alpha_list,
                beta_list=beta_list
            )

    def _build_rho_DE_4D(self):
        """Build rho_DE(a) for 4D temporal quantum multiverse model."""
        from .foreign_matter import get_rho_DE_temporal_func
        return get_rho_DE_temporal_func(
            alpha_de=self.alpha,
            a_contact=self.a_contact,
            beta_de=self.beta,
            smoothness=self.params.get('smoothness', 10.0),
            interference_scale=self.params.get('interference_scale', 0.2)
        )

    def _build_rho_DM_func(self):
        """Build rho_DM_foreign(a) function for the current geometry."""
        from .foreign_matter import get_rho_DM_func
        return get_rho_DM_func(
            geometry=self.geometry,
            alpha_dm=self.alpha_dm,
            a_contact=self.a_contact,
            beta_dm=self.beta_dm,
            smoothness=self.params.get('smoothness', 10.0),
            overdensity_factor=self.params.get('overdensity_factor', 1.0),
            tunneling_scale=self.params.get('tunneling_scale', 0.15),
            time_dilation_enhancement=self.params.get('time_dilation_enhancement', 0.0)
        )

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
        print(f"Geometry: {self.geometry}")
        print(f"Foreign DM: {'Yes' if self.include_foreign_dm else 'No'}")
        print("Parameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v}")
        if self.results:
            print(f"\n  H0 = {self.results['H_km_s_Mpc'][0]:.2f} km/s/Mpc")
            print(f"  Omega_m (total) = {self.results['Omega_m'][0]:.4f}")
            print(f"  Omega_m (local) = {self.results['Omega_m_local'][0]:.4f}")
            print(f"  Omega_DM (foreign) = {self.results['Omega_DM_foreign'][0]:.4f}")
            print(f"  Omega_DE = {self.results['Omega_DE'][0]:.4f}")
        print("=" * 60)
"""
MCMC parameter estimation for the merging universes model.
Uses the emcee package to sample the posterior distribution
of the model parameters given cosmological data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import emcee
from model import config
from model.merger_model import MergerModel
from fitting.likelihood import total_log_likelihood, model_vs_lcdm_chi2, bic, aic

# Default parameter bounds for MCMC
PARAM_BOUNDS = {
    'alpha': (1e-6, 1.0),
    'beta': (0.5, 5.0),
    'd_init': (0.01, 1.0),
    'v_init': (0.01, 0.5),
    'V_crit': (0.01, 0.5),
    'H0_univ': (60.0, 80.0),
    'Omega_m_univ': (0.1, 0.5),
}


class MergerMCMC:
    """
    MCMC sampler for the merging universes model.
    Fits parameters: alpha, beta, d_init
    """

    def __init__(self, sne_data, jwst_data=None, hz_data=None, growth_data=None,
                 free_params=None, n_walkers=32, n_steps=2000, n_burn=500):
        self.sne_data = sne_data
        self.jwst_data = jwst_data
        self.hz_data = hz_data
        self.growth_data = growth_data
        self.free_params = free_params or ['alpha', 'beta', 'd_init']
        self.n_params = len(self.free_params)
        self.n_walkers = n_walkers if n_walkers > 2 * self.n_params else 2 * self.n_params
        self.n_steps = n_steps
        self.n_burn = n_burn
        self.fixed_params = config.MERGER_DEFAULTS.copy()
        self.sampler = None
        self.chain = None
        self.best_params = None
        self.best_log_like = -np.inf

    def _params_to_dict(self, theta):
        p = self.fixed_params.copy()
        for i, name in enumerate(self.free_params):
            p[name] = theta[i]
        return p

    def _check_bounds(self, theta):
        for i, name in enumerate(self.free_params):
            if name in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[name]
                if theta[i] < lo or theta[i] > hi:
                    return False
        return True

    def log_prior(self, theta):
        if not self._check_bounds(theta):
            return -np.inf
        for i, name in enumerate(self.free_params):
            if name == 'alpha':
                if theta[i] <= 0:
                    return -np.inf
                return -np.log(theta[i])
        return 0.0

    def log_likelihood(self, theta):
        if not self._check_bounds(theta):
            return -np.inf
        params = self._params_to_dict(theta)
        try:
            model = MergerModel(params)
            model.solve()
            log_like, components = total_log_likelihood(
                model, self.sne_data, self.jwst_data,
                self.hz_data, self.growth_data
            )
            if log_like > self.best_log_like:
                self.best_log_like = log_like
                self.best_params = theta.copy()
                self.best_components = components
            return log_like
        except Exception:
            return -np.inf

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def _initialize_walkers(self):
        ndim = self.n_params
        pos = np.zeros((self.n_walkers, ndim))
        for i, name in enumerate(self.free_params):
            default = self.fixed_params.get(name, PARAM_BOUNDS.get(name, (0, 1))[0])
            if isinstance(default, tuple):
                default = default[0]
            lo, hi = PARAM_BOUNDS.get(name, (default * 0.1, default * 10))
            if name == 'alpha':
                pos[:, i] = np.exp(np.random.uniform(np.log(lo), np.log(hi), self.n_walkers))
            else:
                pos[:, i] = np.random.uniform(lo, hi, self.n_walkers)
        return pos

    def run(self, progress=True):
        ndim = self.n_params
        pos = self._initialize_walkers()
        print(f"Running MCMC with {self.n_walkers} walkers, {self.n_steps} steps")
        print(f"Free parameters: {self.free_params}")
        self.sampler = emcee.EnsembleSampler(self.n_walkers, ndim, self.log_posterior)
        self.sampler.run_mcmc(pos, self.n_steps, progress=progress)
        self.chain = self.sampler.get_chain(discard=self.n_burn, flat=True)
        max_idx = np.argmax(self.sampler.get_log_prob(discard=self.n_burn, flat=True))
        self.best_params = self.chain[max_idx]
        print(f"\nMCMC complete. Best log-likelihood: {self.best_log_like:.2f}")
        for i, name in enumerate(self.free_params):
            print(f"  {name}: {self.best_params[i]:.6f}")
        return {
            'chain': self.chain,
            'best_params': self.best_params,
            'best_log_like': self.best_log_like,
            'sampler': self.sampler,
        }

    def get_best_model(self):
        if self.best_params is None:
            return None
        params = self._params_to_dict(self.best_params)
        model = MergerModel(params)
        model.solve()
        model.solve_lcdm()
        return model

    def parameter_summary(self):
        if self.chain is None:
            print("No chain available.")
            return
        print(f"\n{'='*60}")
        print("MCMC Parameter Summary")
        print(f"{'='*60}")
        print(f"{'Parameter':<15} {'Median':<12} {'16th':<12} {'84th':<12} {'Best':<12}")
        print(f"{'-'*60}")
        for i, name in enumerate(self.free_params):
            median = np.median(self.chain[:, i])
            p16 = np.percentile(self.chain[:, i], 16)
            p84 = np.percentile(self.chain[:, i], 84)
            best = self.best_params[i]
            print(f"{name:<15} {median:<12.4f} {p16:<12.4f} {p84:<12.4f} {best:<12.4f}")
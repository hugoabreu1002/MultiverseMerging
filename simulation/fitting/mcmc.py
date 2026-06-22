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
import multiprocessing
import os
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
                 free_params=None, n_walkers=32, n_steps=2000, n_burn=500,
                 multi_universe=False, N_universes=1, hierarchical=False,
                 n_procs=None):
        self.sne_data = sne_data
        self.jwst_data = jwst_data
        self.hz_data = hz_data
        self.growth_data = growth_data
        self.multi_universe = multi_universe
        self.N_universes = N_universes
        self.hierarchical = hierarchical
        self.n_procs = n_procs
        self.free_params = free_params or ['alpha', 'beta', 'd_init']
        # Expand free parameters for multi-universe case
        if self.multi_universe and self.N_universes > 1:
            # remove any existing alpha/a_contact placeholders
            base = [p for p in self.free_params if not p.startswith('alpha_') and not p.startswith('a_contact_')]
            extra = []
            for i in range(self.N_universes):
                extra.append(f'alpha_{i}')
                extra.append(f'a_contact_{i}')
            self.free_params = base + extra
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
        alpha_list = []
        a_contact_list = []
        for i, name in enumerate(self.free_params):
            if name.startswith('alpha_'):
                alpha_list.append(theta[i])
            elif name.startswith('a_contact_'):
                a_contact_list.append(theta[i])
            else:
                p[name] = theta[i]
        if alpha_list:
            p['alpha_list'] = alpha_list
        if a_contact_list:
            p['a_contact_list'] = a_contact_list
        return p

    def _check_bounds(self, theta):
        for i, name in enumerate(self.free_params):
            if name in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[name]
            else:
                # Patterned bounds for per-universe params
                if name.startswith('alpha_'):
                    lo, hi = PARAM_BOUNDS.get('alpha', (1e-6, 1.0))
                elif name.startswith('a_contact_'):
                    lo, hi = (0.0, 1.0)
                else:
                    lo, hi = (None, None)
            if lo is not None and hi is not None:
                if theta[i] < lo or theta[i] > hi:
                    return False
        return True

    def log_prior(self, theta):
        if not self._check_bounds(theta):
            return -np.inf
        # Apply simple priors: log-uniform for alpha-like params
        lp = 0.0
        # Base alpha scale for hierarchical priors
        base_alpha = self.fixed_params.get('alpha', 1.0) / max(1, self.N_universes)
        for i, name in enumerate(self.free_params):
            if name == 'alpha' or name.startswith('alpha_'):
                if theta[i] <= 0:
                    return -np.inf
                # log-uniform prior
                lp += -np.log(theta[i])
                # Hierarchical soft constraint: alpha_i ~ LogNormal(log(base_alpha), 0.5)
                if self.hierarchical and name.startswith('alpha_'):
                    sigma = 0.5
                    mu = np.log(base_alpha + 1e-30)
                    lp += -0.5 * ((np.log(theta[i]) - mu) / sigma) ** 2
            if name.startswith('a_contact_') and self.hierarchical:
                # Soft Gaussian prior around global a_contact
                a0 = self.fixed_params.get('a_contact', 0.5)
                sigma_ac = 0.2
                lp += -0.5 * ((theta[i] - a0) / sigma_ac) ** 2
        return lp

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
            # Determine bounds
            if name in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[name]
            elif name.startswith('alpha_'):
                lo, hi = PARAM_BOUNDS.get('alpha', (1e-6, 1.0))
            elif name.startswith('a_contact_'):
                lo, hi = (0.0, 1.0)
            else:
                # Fallback
                lo, hi = (0.0, 1.0)
            if name == 'alpha' or name.startswith('alpha_'):
                pos[:, i] = np.exp(np.random.uniform(np.log(lo), np.log(hi), self.n_walkers))
            else:
                pos[:, i] = np.random.uniform(lo, hi, self.n_walkers)
        return pos

    def run(self, progress=True):
        ndim = self.n_params
        pos = self._initialize_walkers()
        print(f"Running MCMC with {self.n_walkers} walkers, {self.n_steps} steps")
        print(f"Free parameters: {self.free_params}")
        # Use multiprocessing pool if requested (or auto-detect)
        pool = None
        use_pool = False
        if self.n_procs is None:
            # Auto choose a reasonable number of processes
            try:
                cpu = multiprocessing.cpu_count()
            except Exception:
                cpu = 1
            self.n_procs = min(self.n_walkers, max(1, cpu))
        if self.n_procs and self.n_procs > 1:
            use_pool = True

        if use_pool:
            print(f"Using multiprocessing pool with {self.n_procs} processes")
            ctx = multiprocessing.get_context('fork') if hasattr(multiprocessing, 'get_context') else multiprocessing
            pool = ctx.Pool(processes=self.n_procs)
            self.sampler = emcee.EnsembleSampler(self.n_walkers, ndim, self.log_posterior, pool=pool)
            try:
                self.sampler.run_mcmc(pos, self.n_steps, progress=progress)
            finally:
                pool.close()
                pool.join()
        else:
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
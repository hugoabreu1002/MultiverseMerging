"""
JWST/MAST data query module.
Queries the MAST API for JWST observations, including:
- High-redshift galaxy candidates (z > 6)
- Galaxy stellar masses and photometric redshifts
- NIRCam imaging and spectroscopy data

Uses the MAST API v0 endpoint: https://mast.stsci.edu/api/v0/invoke
"""

import json
import numpy as np
import requests
from pathlib import Path
from scipy.integrate import cumulative_trapezoid as cumtrapz

# Import config for constants
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import config

# Cache directory for query results
CACHE_DIR = Path(__file__).parent.parent / 'results' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MAST API endpoint
MAST_URL = "https://mast.stsci.edu/api/v0/invoke"

# Timeout for requests (seconds)
TIMEOUT = 60


def _mast_request(service, params, format='json', page=1, page_size=1000):
    """
    Make a request to the MAST API.
    """
    request_payload = {
        'service': service,
        'params': params,
        'format': format,
        'page': page,
        'pageSize': page_size,
    }
    response = requests.get(
        MAST_URL,
        params={'request': json.dumps(request_payload)},
        timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _cache_or_query(cache_name, service, params, page_size=1000):
    """Check cache first, then query MAST if needed."""
    cache_path = CACHE_DIR / f"{cache_name}.npy"
    if cache_path.exists():
        print(f"  Loading cached: {cache_name}")
        return np.load(cache_path, allow_pickle=True).item()
    print(f"  Querying MAST: {cache_name}")
    try:
        result = _mast_request(service, params, page_size=page_size)
        data = result.get('data', [])
        if data:
            np.save(cache_path, data)
            print(f"  Retrieved {len(data)} results")
        return data
    except Exception as e:
        print(f"  MAST query failed: {e}")
        return []


def query_jwst_field(ra=53.0, dec=-27.0, radius=0.5, instrument='NIRCAM',
                     use_cache=True):
    """
    Query JWST observations in a given field.

    Parameters
    ----------
    ra, dec : float
        Right ascension and declination (degrees)
    radius : float
        Search radius (degrees)
    instrument : str
        Instrument name filter ('NIRCAM', 'NIRSPEC', 'MIRI')
    use_cache : bool
        Use cached results if available

    Returns
    -------
    list of dict
        Matching observations
    """
    cache_name = f"jwst_field_{ra:.1f}_{dec:.1f}_r{radius:.1f}"

    if use_cache:
        cache_path = CACHE_DIR / f"{cache_name}.npy"
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=True).item()

    results = _cache_or_query(cache_name, 'Mast.Caom.Cone',
                              {'ra': ra, 'dec': dec, 'radius': radius},
                              page_size=500)

    if isinstance(results, dict):
        results = [results]

    filtered = []
    for obs in results:
        if isinstance(obs, dict):
            obs_col = obs.get('obs_collection', '')
            inst = obs.get('instrument_name', '')
            if 'JWST' in obs_col and instrument.upper() in inst.upper():
                filtered.append(obs)

    if use_cache:
        np.save(CACHE_DIR / f"{cache_name}_filtered.npy", filtered)

    print(f"  Found {len(filtered)} JWST {instrument} observations in field")
    return filtered


def query_jwst_ceers_field():
    """CEERS field: RA=214.9, Dec=52.95"""
    return query_jwst_field(ra=214.9, dec=52.95, radius=0.3)


def query_jwst_jades_field():
    """JADES field (GOODS-S): RA=53.0, Dec=-27.8"""
    return query_jwst_field(ra=53.0, dec=-27.8, radius=0.3)


def query_jwst_glass_field():
    """GLASS field: RA=3.5, Dec=-30.4"""
    return query_jwst_field(ra=3.5, dec=-30.4, radius=0.2)


def load_simulated_jwst_highz_data():
    """
    Simulated high-z galaxy mass function data based on JWST results
    from Labbe et al. 2023 (Nature), Robertson et al. 2023, etc.
    """
    data = {'z': [], 'logM': [], 'log_phi': []}

    # z ~ 8-9
    for z in [8.5, 9.0]:
        data['z'].extend([z] * 5)
        data['logM'].extend([9.0, 9.5, 10.0, 10.5, 11.0])
        data['log_phi'].extend([-3.5, -4.0, -4.8, -5.5, -6.5])

    # z ~ 10-12
    for z in [10.5, 11.0]:
        data['z'].extend([z] * 3)
        data['logM'].extend([9.0, 9.5, 10.0])
        data['log_phi'].extend([-4.0, -5.0, -6.0])

    # z ~ 13-15
    data['z'].extend([13.5, 14.0, 15.0])
    data['logM'].extend([8.5, 9.0, 8.5])
    data['log_phi'].extend([-4.5, -5.5, -5.0])

    return data


def load_pantheon_plus_data():
    """
    Simulated Pantheon+ SN Ia distance modulus data.
    For real usage, download from: https://github.com/PantheonPlusSH0ES/DataRelease
    """
    np.random.seed(42)

    z = np.concatenate([
        np.linspace(0.02, 0.1, 50),
        np.linspace(0.1, 0.5, 100),
        np.linspace(0.5, 1.0, 80),
        np.linspace(1.0, 2.3, 50),
    ])

    # ΛCDM with Planck 2018
    Om = config.PLANCK_2018['Omega_m']
    Or = config.PLANCK_2018['Omega_r']
    OL = config.PLANCK_2018['Omega_Lambda']
    H0 = config.PLANCK_2018['H0']

    def H_lcdm(z):
        return H0 * np.sqrt(Om * (1 + z)**3 + Or * (1 + z)**4 + OL)

    Hz = H_lcdm(z) * config.km_s_Mpc_to_1_per_s
    integrand = config.c / Hz
    dc = cumtrapz(integrand, z, initial=0)
    dl = (1 + z) * dc
    dl = np.maximum(dl, 1e-10)
    mu_true = 5.0 * np.log10(dl) + 25.0

    mu_err = 0.1 + 0.02 * z
    mu_obs = mu_true + np.random.normal(0, mu_err)

    return {'z': z, 'mu': mu_obs, 'mu_err': mu_err, 'mu_true': mu_true}
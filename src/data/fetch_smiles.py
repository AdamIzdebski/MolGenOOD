""" Fetch SMILES from ZINC database.
"""

import random
import requests
from typing import List
import time
import gzip
import io
from urllib.parse import urlparse, urlunparse

import numpy as np
from tqdm import tqdm
from src.data.constants import ZINC_DRUGLIKE_TRANCHES
from src.data.chemistry import standardize_smiles

UA = {"User-Agent": "Mozilla/5.0"}


def _fetch_zinc_tranche(tranche: str, folder: str, timeout: int = 20, tries: int = 3, sleep: float = 1.5) -> list:
    """ Fetch a ZINC tranche and return a list of the first column entries.
    """
    base = f"https://files.docking.org/2D/{folder}/{tranche}"
    urls = [base + ".txt", base + ".txt.gz"]
    last_err = None
    
    for url in urls:
        # Force https
        u = list(urlparse(url))
        u[0] = "https"
        url = urlunparse(u)
        
        for attempt in range(tries):
            try:
                r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True, stream=True)
                if r.status_code == 200:
                    data = r.content
                    if url.endswith(".gz"):
                        data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
                    result = [line.split("\t", 1)[0]
                              for line in data.decode("utf-8", errors="replace").splitlines()
                              if line.strip()]
                    return result[1:]  # Skip header, return list instead of yielding
                elif r.status_code in (403, 429, 500, 502, 503, 504):
                    time.sleep(sleep * (attempt + 1))
                    continue
                else:
                    r.raise_for_status()
            except Exception as e:
                last_err = e
                time.sleep(sleep * (attempt + 1))
    
    raise RuntimeError(f"Failed to fetch tranche {tranche}: {last_err}")

def fetch_smiles(seed: int, n_samples: int = None) -> List[str]:

    # Fetch data
    sampled_sequences = _fetch_zinc(n_samples, seed)

    # Filter out None values (failed standardizations)
    standardized = [standardize_smiles(smiles) for smiles in sampled_sequences]
    valid_smiles = [s for s in standardized if s is not None]
    
    print("Returning", len(valid_smiles), " out of ", len(sampled_sequences), "SMILES, after standardization.")
    return valid_smiles

def _fetch_zinc(n_samples: int = None, seed: int = 0) -> np.ndarray:
    """Download SMILES from ZINC database.
    
    Args:
        n_samples: Number of SMILES to download. If None, downloads from all tranches.
        seed: Random seed for reproducibility.
        
    Returns:
        Array of SMILES strings.
    """
    random.seed(seed)

    if n_samples is None:
        # If no specific number requested, use all tranches
        sampled_tranches = ZINC_DRUGLIKE_TRANCHES
        target_samples_per_tranche = None  # Take all available
    else:
        # Calculate how many tranches we need and samples per tranche
        # Aim for roughly equal distribution across tranches
        ESTIMATED_SAMPLES_PER_TRANCHE = 100  # Conservative estimate
        NUM_TRANCHES = min(max(1, n_samples // ESTIMATED_SAMPLES_PER_TRANCHE), len(ZINC_DRUGLIKE_TRANCHES))
        target_samples_per_tranche = (n_samples // NUM_TRANCHES) + 1
        
        sampled_tranches = random.sample(ZINC_DRUGLIKE_TRANCHES, NUM_TRANCHES)
    
    sampled_sequences = []
    successful_tranches = 0
    failed_tranches = 0

    # Create progress bar with better formatting
    pbar = tqdm(sampled_tranches, desc="Downloading ZINC tranches", 
                unit="tranche", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for _tranche_code in pbar:
        try:
            _sampled_sequences = _fetch_zinc_tranche(_tranche_code, _tranche_code[:2])
            
            # If we have a target number of samples per tranche, sample from the results
            if target_samples_per_tranche is not None and len(_sampled_sequences) > target_samples_per_tranche:
                _sampled_sequences = random.sample(_sampled_sequences, target_samples_per_tranche)
            
            sampled_sequences.extend(_sampled_sequences)
            successful_tranches += 1
            
            # Update progress bar description with current stats
            pbar.set_postfix({
                'molecules': len(sampled_sequences),
                'success': successful_tranches, 
                'failed': failed_tranches
            })
            
            # Stop early if we have enough samples
            if n_samples is not None and len(sampled_sequences) >= n_samples:
                break
                
        except Exception as e:
            failed_tranches += 1
            tqdm.write(f"Error for tranche {_tranche_code}: {e}")
            # Update progress bar even on failure
            pbar.set_postfix({
                'molecules': len(sampled_sequences),
                'success': successful_tranches, 
                'failed': failed_tranches
            })

    tqdm.write(f"Download completed: {successful_tranches} successful, {failed_tranches} failed tranches")
    tqdm.write(f"Retrieved {len(sampled_sequences)} SMILES")
    
    # Return the requested number of samples
    if n_samples is not None and len(sampled_sequences) > n_samples:
        sampled_sequences = random.sample(sampled_sequences, n_samples)
        
    return np.array(sampled_sequences)

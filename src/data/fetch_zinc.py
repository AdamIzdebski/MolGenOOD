"""ZINC tranche downloader: concise, thread-safe, with minimal progress reporting."""

import gzip
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from urllib.parse import urlparse, urlunparse

import requests
from tqdm import tqdm

from src.data.constants import ZINC_DRUGLIKE_TRANCHES


# Networking defaults (balanced for multithreading)
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
RETRY_STATUS_CODES = (403, 429, 500, 502, 503, 504)
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES_PER_URL = 4


def _force_https(url: str) -> str:
    parts = list(urlparse(url))
    parts[0] = "https"
    return urlunparse(parts)


def _fetch_tranche(
    tranche: str,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES_PER_URL,
    limit: int | None = None,
    stop: Event | None = None,
) -> tuple[list[str], int]:
    """Stream a tranche and return (SMILES, total_rows) using reservoir sampling if limit is set."""
    base = f"https://files.docking.org/2D/{tranche[:2]}/{tranche}"
    urls = [_force_https(base + ".txt.gz"), _force_https(base + ".txt")]  # prefer gz first

    last_error: Exception | None = None
    for url in urls:
        for attempt in range(retries):
            if stop is not None and stop.is_set():
                return [], 0
            try:
                with requests.get(
                    url, headers=USER_AGENT, timeout=(timeout, timeout), allow_redirects=True, stream=True
                ) as resp:
                    if resp.status_code == 200:
                        header_skipped = False
                        picked: list[str] = []
                        total = 0

                        if url.endswith(".gz"):
                            decoder = gzip.GzipFile(fileobj=resp.raw)
                            for bline in decoder:
                                if stop is not None and stop.is_set():
                                    break
                                line = bline.decode("utf-8", errors="replace").strip()
                                if not line:
                                    continue
                                if not header_skipped:
                                    header_skipped = True
                                    continue
                                total += 1
                                smi = line.split("\t", 1)[0]
                                if limit is None:
                                    picked.append(smi)
                                else:
                                    if len(picked) < limit:
                                        picked.append(smi)
                                    else:
                                        j = random.randint(0, total - 1)
                                        if j < limit:
                                            picked[j] = smi
                        else:
                            for line in resp.iter_lines(decode_unicode=True):
                                if stop is not None and stop.is_set():
                                    break
                                if line is None:
                                    continue
                                line = line.strip()
                                if not line:
                                    continue
                                if not header_skipped:
                                    header_skipped = True
                                    continue
                                total += 1
                                smi = line.split("\t", 1)[0]
                                if limit is None:
                                    picked.append(smi)
                                else:
                                    if len(picked) < limit:
                                        picked.append(smi)
                                    else:
                                        j = random.randint(0, total - 1)
                                        if j < limit:
                                            picked[j] = smi

                        return picked, total

                    if resp.status_code in RETRY_STATUS_CODES:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    resp.raise_for_status()
            except Exception as e:  # noqa: BLE001
                last_error = e
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch tranche {tranche}: {last_error}")


def _plan_tranches(
    n_samples: int | None, seed: int, estimated_samples_per_tranche: int
) -> tuple[list[str], int | None]:
    if n_samples is None:
        return list(ZINC_DRUGLIKE_TRANCHES), None
    random.seed(seed)
    num_tranches = min(
        (n_samples // estimated_samples_per_tranche) + 1, len(ZINC_DRUGLIKE_TRANCHES)
    )
    target_per_tranche = (n_samples // num_tranches) + 1
    return random.sample(ZINC_DRUGLIKE_TRANCHES, num_tranches), target_per_tranche


def fetch_zinc(
    n_samples: int | None = None,
    seed: int = 42,
    max_workers: int = 4,
    estimated_samples_per_tranche: int = 10,
) -> list[str]:
    """Public API: Fetch SMILES strings from ZINC.

    - Concurrent tranche downloads with retries and gzip support
    - Minimal tqdm reporting (current tranche stats and overall counts)
    - Uses an estimated average samples per tranche to plan parallel work
    """
    start_time = time.time()
    sampled_tranches, target_per_tranche = _plan_tranches(
        n_samples, seed, estimated_samples_per_tranche
    )

    collected: list[str] = []
    successes = 0
    failures = 0
    lock = Lock()
    stop_event = Event()

    pbar = tqdm(total=len(sampled_tranches), desc="Downloading ZINC tranches", unit="tranche",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def worker(tranche_code: str) -> tuple[str, list[str], int, bool, str | None]:
        try:
            if stop_event.is_set():
                return tranche_code, [], 0, False, "stopped"
            seqs, total = _fetch_tranche(tranche_code, limit=target_per_tranche, stop=stop_event)
            return tranche_code, seqs, total, True, None
        except Exception as e:  # noqa: BLE001
            return tranche_code, [], 0, False, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, t) for t in sampled_tranches]
        for fut in as_completed(futures):
            tranche_code, seqs, original, ok, err = fut.result()
            with lock:
                if ok:
                    collected.extend(seqs)
                    successes += 1
                    last = f"✓{original}→{len(seqs)}"
                else:
                    failures += 1
                    last = f"✗{tranche_code}"
                pbar.update(1)
                pbar.set_postfix({
                    'molecules': len(collected), 'last': last, 'success': successes, 'failed': failures
                })
                if n_samples is not None and len(collected) >= n_samples:
                    stop_event.set()
                    # Try to cancel not-yet-started futures
                    for f in futures:
                        f.cancel()
                    break
    pbar.close()

    if n_samples is not None and len(collected) > n_samples:
        collected = random.sample(collected, n_samples)
    elapsed = time.time() - start_time
    tqdm.write(
        f"✅ Fetched {len(collected)} molecules in {elapsed:.1f}s from {successes} tranches (failed {failures})."
    )
    return collected



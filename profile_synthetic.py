#!/usr/bin/env python3
"""Profile CompactTree ingestion and/or lookup of a synthetic 3-level nested dict.

Dict shape (unique keys per level):
    {0: 9, 1: 4, 2: <L2>}   (default L2: 173000 for build/lookup, 10000 for serde)

Keys are drawn from N-gram permutations (N=1..7) from corpus.txt.
Values (leaves) are also drawn from the same permutation pool (N=1..7,
~4.6M entries), sampled with replacement so each leaf may differ.

Usage
-----
  python profile_synthetic.py                   # profile build (default)
  python profile_synthetic.py --mode build      # profile build
  python profile_synthetic.py --mode lookup     # profile lookup (~10 s)
  python profile_synthetic.py --mode both       # profile build then lookup
  python profile_synthetic.py --mode serialize  # profile single serialize
  python profile_synthetic.py --mode deserialize # profile single deserialize
  python profile_synthetic.py --mode serde      # profile serialize then deserialize
  python profile_synthetic.py --l2 5000 --mode serde  # override L2 key count
  python profile_synthetic.py --mode lookup --vocab-size 0  # disable LRU, profile _index_uncached
"""

import argparse
import cProfile
import io
import itertools
import os
import pstats
import random
import tempfile
import time
from typing import Optional
from pathlib import Path

from compact_tree import CompactTree

# ---------------------------------------------------------------------------
# N-gram permutation generator
# ---------------------------------------------------------------------------

def _ngram_permutations(words: list[str], max_n: int = 7):
    """Yield ' '.join(perm) for every window of size n=1..max_n in *words*,
    iterating all permutations of each window."""
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            for perm in itertools.permutations(window):
                yield " ".join(perm)


def build_word_pools(corpus_path: Path, max_n: int = 7) -> tuple[list[str], list[str]]:
    """Return (key_pool, value_pool) built from corpus n-gram permutations.

    key_pool  – first TOTAL_KEYS_NEEDED unique n-grams  (deduped)
    value_pool – all n-grams as a list (repetitions allowed, for fast sampling)
    """
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.split()
    print(f"  Corpus: {len(tokens):,} tokens, generating N=1..{max_n} permutations...")

    t0 = time.perf_counter()
    key_set: set[str] = set()
    key_pool: list[str] = []
    value_pool: list[str] = []

    for ng in _ngram_permutations(tokens, max_n):
        value_pool.append(ng)
        if ng not in key_set:
            key_set.add(ng)
            key_pool.append(ng)

    print(f"  Generated {len(value_pool):,} total / {len(key_pool):,} unique n-grams "
          f"in {time.perf_counter() - t0:.3f}s")
    return key_pool, value_pool


# ---------------------------------------------------------------------------
# Dict construction
# ---------------------------------------------------------------------------

# Target stats
L0_KEYS = 9
L1_KEYS = 4
VALUE_POOL_CAP = 10_000   # cap unique leaf values to keep pre-warm tractable


def build_dict(l2_keys: int = 173_000) -> dict:
    """Build the 3-level nested dict using corpus n-gram permutations."""
    corpus_path = Path(__file__).parent / "corpus.txt"

    print(f"Building vocabulary from corpus n-gram permutations (N=1..7)...")
    key_pool, value_pool = build_word_pools(corpus_path, max_n=7)

    need_keys = L0_KEYS + L1_KEYS + l2_keys
    if len(key_pool) < need_keys:
        raise ValueError(
            f"Not enough unique n-grams ({len(key_pool):,}) for "
            f"{need_keys:,} required keys"
        )

    l0 = key_pool[:L0_KEYS]
    l1 = key_pool[L0_KEYS : L0_KEYS + L1_KEYS]
    l2 = key_pool[L0_KEYS + L1_KEYS : L0_KEYS + L1_KEYS + l2_keys]

    # Cap the value pool so the number of unique leaf values stays tractable
    value_pool = value_pool[:VALUE_POOL_CAP]

    print(f"  L0 keys: {len(l0)}, L1 keys: {len(l1)}, "
          f"L2 keys: {len(l2):,}, value pool: {len(value_pool):,}")

    rng = random.Random(99)
    t1 = time.perf_counter()
    print("Building nested dict...")
    d: dict = {}
    for k0 in l0:
        d[k0] = {}
        for k1 in l1:
            inner = {k2: rng.choice(value_pool) for k2 in l2}
            d[k0][k1] = inner

    total_leaves = L0_KEYS * L1_KEYS * l2_keys
    print(f"  Dict built in {time.perf_counter() - t1:.3f}s  "
          f"({total_leaves:,} leaf entries)")
    return d


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_ingestion(d: dict, vocabulary_size: Optional[int] = None) -> CompactTree:
    """Profile CompactTree.from_dict(d) and print a summary."""
    # Estimate unique keys and values to size the LRU cache exactly.
    all_keys: set[str] = set()
    all_values: set[str] = set()
    def _walk(node: dict) -> None:
        for k, v in node.items():
            all_keys.add(k)
            if isinstance(v, dict):
                _walk(v)
            else:
                all_values.add(str(v))
    _walk(d)
    vocab_hint = (
        f"vocabulary_size={vocabulary_size} (cache DISABLED — every lookup calls _index_uncached)"
        if vocabulary_size == 0
        else f"vocabulary_size={vocabulary_size!r} (auto-sized to vocab)"
        if vocabulary_size is not None
        else f"vocabulary_size=None (auto={len(all_keys) + len(all_values):,})"
    )
    print(f"  Unique keys: {len(all_keys):,}, unique values: {len(all_values):,} "
          f"-> {vocab_hint}")

    profiler = cProfile.Profile()

    print("\nProfiling CompactTree.from_dict() ...")
    wall_start = time.perf_counter()
    profiler.enable()
    tree = CompactTree.from_dict(d, vocabulary_size=vocabulary_size)
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start
    print(f"  Wall time: {wall_elapsed:.3f}s")
    print(f"  Top-level keys in tree: {len(tree)}")

    # Capture stats
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()

    for sort_key, title, n in [
        ("cumulative", "CUMULATIVE TIME (top 30)", 30),
        ("tottime",    "TOTAL (self) TIME (top 30)", 30),
        ("calls",      "MOST-CALLED FUNCTIONS (top 20)", 20),
    ]:
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
        buf.truncate(0)
        buf.seek(0)
        stats.sort_stats(sort_key)
        stats.print_stats(n)
        print(buf.getvalue())

    return tree


# ---------------------------------------------------------------------------
# Shared stats helper
# ---------------------------------------------------------------------------

def _print_profile_stats(
    profiler: cProfile.Profile,
    cum_n: int = 20,
    tot_n: int = 20,
    calls_n: int = 15,
) -> None:
    """Print cumulative, self-time, and most-called tables from *profiler*."""
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    for sort_key, title, n in [
        ("cumulative", f"CUMULATIVE TIME (top {cum_n})",    cum_n),
        ("tottime",    f"TOTAL (self) TIME (top {tot_n})",  tot_n),
        ("calls",      f"MOST-CALLED FUNCTIONS (top {calls_n})", calls_n),
    ]:
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
        buf.truncate(0)
        buf.seek(0)
        stats.sort_stats(sort_key)
        stats.print_stats(n)
        print(buf.getvalue())


# ---------------------------------------------------------------------------
# Lookup profiling
# ---------------------------------------------------------------------------

def profile_lookup(tree: CompactTree, d: dict,
                   duration: float = 10.0,
                   miss_ratio: float = 0.1,
                   use_get_path: bool = False,
                   use_dict: bool = False) -> None:
    """Profile random leaf lookups on *tree* (or *d*) for approximately *duration* seconds.

    Key generation is O(1) per lookup: precompute the small key lists for
    each level, then use ``rng.randrange(len(keys))`` + direct list indexing
    to pick a key.  No path list is precomputed.

    ``miss_ratio`` fraction of lookups intentionally use a key from the wrong
    level (guaranteed miss) to exercise the KeyError / __contains__ path.

    When *use_dict* is True the benchmark runs against the plain Python dict
    *d* instead of the CompactTree, giving a direct baseline comparison.
    """
    # Extract key lists once — tiny: 9, 4, 173K entries.
    l0_keys = list(d.keys())
    # All L0 children have the same L1 keys; sample from the first.
    l1_keys = list(next(iter(d.values())).keys())
    # All L1 children share the same L2 key set; sample from d[l0][l1].
    l2_keys = list(next(iter(next(iter(d.values())).values())).keys())
    n0, n1, n2 = len(l0_keys), len(l1_keys), len(l2_keys)
    print(f"\nKey lists: L0={n0}, L1={n1}, L2={n2:,}")

    # Warmup: prime lru_cache on all three key levels (skip if cache/dict mode).
    if use_dict:
        print("  Using plain Python dict — no warmup needed")
    else:
        cache_disabled = getattr(tree, '_key_vocab_size', None) == 0
        if cache_disabled:
            print("  Cache disabled (vocab_size=0) — skipping warmup, profiling _index_uncached directly")
        else:
            for _ in range(2_000):
                _ = tree[l0_keys[_ % n0]]
            print("  Warmed up")

    rng = random.Random(42)
    profiler = cProfile.Profile()
    CHECK_INTERVAL = 1_000

    print(f"\nProfiling random leaf lookups for ~{duration:.0f}s wall clock "
          f"({miss_ratio*100:.0f}% misses) ...")
    wall_start = time.perf_counter()
    n_iters = 0
    profiler.enable()
    while True:
        for _ in range(CHECK_INTERVAL):
            k0 = l0_keys[rng.randrange(n0)]
            k1 = l1_keys[rng.randrange(n1)]
            k2 = l2_keys[rng.randrange(n2)]
            if rng.random() < miss_ratio:
                # Guaranteed miss: swap k2 for a key from the wrong level.
                k2 = l0_keys[rng.randrange(n0)]
                try:
                    if use_dict:
                        _ = d[k0][k1][k2]
                    elif use_get_path:
                        _ = tree.get_path(k0, k1, k2)
                    else:
                        _ = tree[k0][k1][k2]
                except KeyError:
                    pass
            else:
                if use_dict:
                    _ = d[k0][k1][k2]
                elif use_get_path:
                    _ = tree.get_path(k0, k1, k2)
                else:
                    _ = tree[k0][k1][k2]
        n_iters += CHECK_INTERVAL
        if time.perf_counter() - wall_start >= duration:
            break
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    actual_rate = n_iters / wall_elapsed
    ns_per = 1e9 / actual_rate
    if ns_per < 1000:
        time_str = f"{ns_per:.1f} ns/lookup"
    else:
        time_str = f"{ns_per / 1000:.3f} µs/lookup"
    print(f"  Wall time: {wall_elapsed:.3f}s  "
          f"({actual_rate:,.0f} lookups/s,  "
          f"{time_str})")

    _print_profile_stats(profiler)


# ---------------------------------------------------------------------------
# Serialize / deserialize profiling
# ---------------------------------------------------------------------------

def profile_serialize(tree: CompactTree) -> str:
    """Profile ``tree.serialize()`` (uncompressed) for 10 seconds.

    Returns the temp file path so the caller can feed it to
    :func:`profile_deserialize`.
    """
    with tempfile.NamedTemporaryFile(suffix=".ctree", delete=False) as _f:
        tmp_path = _f.name

    # Warmup: also ensures the file exists and fsspec imports are loaded.
    tree.serialize(tmp_path)
    file_size = os.path.getsize(tmp_path)

    profiler = cProfile.Profile()
    CHECK_INTERVAL = 10
    DURATION = 10.0

    print(f"\nProfiling serialize() (uncompressed) for ~10s wall clock ...")
    print(f"  File size: {file_size / 1_048_576:.2f} MiB")
    wall_start = time.perf_counter()
    n_iters = 0
    profiler.enable()
    while True:
        for _ in range(CHECK_INTERVAL):
            tree.serialize(tmp_path)
        n_iters += CHECK_INTERVAL
        if time.perf_counter() - wall_start >= DURATION:
            break
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    rate = n_iters / wall_elapsed
    print(f"  Wall time: {wall_elapsed:.3f}s  "
          f"({rate:,.1f} serialize/s,  {1e3 / rate:.2f} ms/serialize)")

    _print_profile_stats(profiler)
    return tmp_path


def profile_deserialize(tmp_path: str) -> None:
    """Profile ``CompactTree(path)`` (uncompressed) for 10 seconds."""
    # Warmup.
    _ = CompactTree(tmp_path)

    profiler = cProfile.Profile()
    CHECK_INTERVAL = 10
    DURATION = 10.0

    print(f"\nProfiling deserialize() (uncompressed) for ~10s wall clock ...")
    wall_start = time.perf_counter()
    n_iters = 0
    profiler.enable()
    while True:
        for _ in range(CHECK_INTERVAL):
            _ = CompactTree(tmp_path)
        n_iters += CHECK_INTERVAL
        if time.perf_counter() - wall_start >= DURATION:
            break
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    rate = n_iters / wall_elapsed
    print(f"  Wall time: {wall_elapsed:.3f}s  "
          f"({rate:,.1f} deserialize/s,  {1e3 / rate:.2f} ms/deserialize)")

    _print_profile_stats(profiler)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CompactTree")
    parser.add_argument(
        "--mode",
        choices=["build", "lookup", "both", "serialize", "deserialize", "serde"],
        default="build",
        help="What to profile (default: build)",
    )
    parser.add_argument(
        "--lookup-duration",
        type=float,
        default=10.0,
        metavar="SECS",
        help="How many seconds to run lookup profiling (default: 10)",
    )
    parser.add_argument(
        "--l2",
        type=int,
        default=None,
        metavar="N",
        help="Number of L2 keys (default: 10000 for serde modes, 173000 otherwise)",
    )
    parser.add_argument(
        "--use-get-path",
        action="store_true",
        default=False,
        dest="use_get_path",
        help="Use tree.get_path(k0,k1,k2) instead of tree[k0][k1][k2] in the lookup benchmark",
    )
    parser.add_argument(
        "--use-dict",
        action="store_true",
        default=False,
        dest="use_dict",
        help="Benchmark the plain Python dict instead of CompactTree (baseline comparison)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        metavar="N",
        dest="vocab_size",
        help="vocabulary_size passed to from_dict (sets lru_cache maxsize). "
             "Use 0 to disable the LRU cache entirely and profile _index_uncached directly.",
    )
    args = parser.parse_args()

    # Resolve L2 size: explicit > mode default > global default
    if args.l2 is not None:
        l2_size = args.l2
    elif args.mode in ("serialize", "deserialize", "serde"):
        l2_size = 10_000
    else:
        l2_size = 173_000

    d = build_dict(l2_keys=l2_size)

    if args.mode in ("build", "both"):
        tree = profile_ingestion(d, vocabulary_size=args.vocab_size)
    else:
        # Build silently for all other modes.
        print("\nBuilding CompactTree (unprofiled)...")
        t0 = time.perf_counter()
        tree = CompactTree.from_dict(d, vocabulary_size=args.vocab_size)
        print(f"  Built in {time.perf_counter() - t0:.3f}s")

    if args.mode in ("lookup", "both"):
        profile_lookup(tree, d, duration=args.lookup_duration,
                       use_get_path=args.use_get_path,
                       use_dict=args.use_dict)

    if args.mode in ("serialize", "serde"):
        tmp = profile_serialize(tree)
        if args.mode == "serde":
            profile_deserialize(tmp)
        os.unlink(tmp)

    if args.mode == "deserialize":
        with tempfile.NamedTemporaryFile(suffix=".ctree", delete=False) as _f:
            tmp = _f.name
        print("\nSerializing once (unprofiled) to temp file...")
        tree.serialize(tmp)
        profile_deserialize(tmp)
        os.unlink(tmp)

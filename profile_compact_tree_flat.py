#!/usr/bin/env python3
"""Profile CompactTreeFlat build, lookup, and serde on a synthetic 3-level nested dict.

Dict shape (unique keys per level):
    {0: 9, 1: 4, 2: <L2>}   (default L2: 173000 for build/lookup, 10000 for serde)

Keys are drawn from N-gram permutations (N=1..7) from corpus.txt.
Leaf values are also drawn from the same permutation pool (~4.6 M entries),
sampled with replacement so each leaf may differ.

Usage
-----
  python profile_compact_tree_flat.py                     # profile build (default)
  python profile_compact_tree_flat.py --mode build        # profile from_dict()
  python profile_compact_tree_flat.py --mode lookup       # profile get_path() (~10 s)
  python profile_compact_tree_flat.py --mode both         # build then lookup
  python profile_compact_tree_flat.py --mode serialize    # profile serialize()
  python profile_compact_tree_flat.py --mode deserialize  # profile load
  python profile_compact_tree_flat.py --mode serde        # serialize + deserialize
  python profile_compact_tree_flat.py --l2 5000 --mode serde
  python profile_compact_tree_flat.py --mode lookup --vocab-size None
  python profile_compact_tree_flat.py --mode both --compare   # also benchmark CompactTree.get_path
"""

from __future__ import annotations

import argparse
import cProfile
import io
import itertools
import os
import pstats
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Generator, Optional

from compact_tree_flat import CompactTreeFlat

# ---------------------------------------------------------------------------
# N-gram permutation generator  (identical to profile_synthetic.py)
# ---------------------------------------------------------------------------

def _ngram_permutations(words: list[str], max_n: int = 7) -> Generator[str, None, None]:
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            for perm in itertools.permutations(window):
                yield " ".join(perm)


def build_word_pools(corpus_path: Path, max_n: int = 7) -> tuple[list[str], list[str]]:
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

L0_KEYS = 9
L1_KEYS = 4
VALUE_POOL_CAP = 10_000


def build_dict(l2_keys: int = 173_000) -> dict[str, dict[str, dict[str, str]]]:
    corpus_path = Path(__file__).parent / "corpus.txt"
    print("Building vocabulary from corpus n-gram permutations (N=1..7)...")
    key_pool, value_pool = build_word_pools(corpus_path, max_n=7)

    need_keys = L0_KEYS + L1_KEYS + l2_keys
    if len(key_pool) < need_keys:
        raise ValueError(
            f"Not enough unique n-grams ({len(key_pool):,}) for "
            f"{need_keys:,} required keys"
        )

    l0 = key_pool[:L0_KEYS]
    l1 = key_pool[L0_KEYS: L0_KEYS + L1_KEYS]
    l2 = key_pool[L0_KEYS + L1_KEYS: L0_KEYS + L1_KEYS + l2_keys]
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
            d[k0][k1] = {k2: rng.choice(value_pool) for k2 in l2}

    total_leaves = L0_KEYS * L1_KEYS * l2_keys
    print(f"  Dict built in {time.perf_counter() - t1:.3f}s  "
          f"({total_leaves:,} leaf entries)")
    return d


# ---------------------------------------------------------------------------
# Shared stats helper
# ---------------------------------------------------------------------------

def _print_profile_stats(
    profiler: cProfile.Profile,
    cum_n: int = 25,
    tot_n: int = 25,
    calls_n: int = 20,
) -> None:
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    for sort_key, title, n in [
        ("cumulative", f"CUMULATIVE TIME (top {cum_n})", cum_n),
        ("tottime",    f"TOTAL (self) TIME (top {tot_n})", tot_n),
        ("calls",      f"MOST-CALLED FUNCTIONS (top {calls_n})", calls_n),
    ]:
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
        buf.truncate(0); buf.seek(0)
        stats.sort_stats(sort_key)
        stats.print_stats(n)
        print(buf.getvalue())


# ---------------------------------------------------------------------------
# Build profiling
# ---------------------------------------------------------------------------

def profile_ingestion(
    d: dict[str, Any],
    vocabulary_size: Optional[int] = 0,
) -> CompactTreeFlat:
    """Profile CompactTreeFlat.from_dict(d) and print a cProfile summary."""
    unique_values: set[str] = set()
    def _count(node: dict[str, Any]) -> None:
        for v in node.values():
            if isinstance(v, dict):
                _count(v)
            else:
                unique_values.add(str(v))
    _count(d)

    vocab_hint = (
        f"vocabulary_size={vocabulary_size} (cache DISABLED — default)"
        if vocabulary_size == 0
        else (
            f"vocabulary_size=None (auto-sized, "
            f"values={len(unique_values):,})"
        )
        if vocabulary_size is None
        else f"vocabulary_size={vocabulary_size!r}"
    )
    print(f"  Unique leaf values: {len(unique_values):,} -> {vocab_hint}")

    profiler = cProfile.Profile()
    print("\nProfiling CompactTreeFlat.from_dict() ...")
    wall_start = time.perf_counter()
    profiler.enable()
    tree = CompactTreeFlat.from_dict(d, vocabulary_size=vocabulary_size)
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start
    print(f"  Wall time: {wall_elapsed:.3f}s")
    print(f"  Total leaf paths: {len(tree):,}")

    _print_profile_stats(profiler)
    return tree


# ---------------------------------------------------------------------------
# Lookup profiling
# ---------------------------------------------------------------------------

def profile_lookup(
    flat_tree: CompactTreeFlat,
    compact_tree: Any,   # CompactTree | None
    d: dict[str, Any],
    duration: float = 10.0,
    miss_ratio: float = 0.1,
    use_dict: bool = False,
) -> None:
    """Profile random leaf lookups for approximately *duration* seconds.

    When *compact_tree* is not None, a second timed loop benchmarks
    ``CompactTree.get_path`` with identical inputs for direct comparison.
    When *use_dict* is True, a third timed loop benchmarks ``d[k0][k1][k2]``
    against the plain Python dict as a baseline.
    """
    l0_keys = list(d.keys())
    l1_keys = list(next(iter(d.values())).keys())
    l2_keys = list(next(iter(next(iter(d.values())).values())).keys())
    n0, n1, n2 = len(l0_keys), len(l1_keys), len(l2_keys)
    print(f"\nKey lists: L0={n0}, L1={n1}, L2={n2:,}")

    def _run_timed(label: str, lookup_fn: Any, miss_fn: Any) -> None:
        rng = random.Random(42)
        profiler = cProfile.Profile()
        CHECK_INTERVAL = 1_000
        print(f"\nProfiling {label} for ~{duration:.0f}s wall clock "
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
                    # Guaranteed miss: swap k2 for an L0 key
                    bad_k2 = l0_keys[rng.randrange(n0)]
                    try:
                        miss_fn(k0, k1, bad_k2)
                    except KeyError:
                        pass
                else:
                    lookup_fn(k0, k1, k2)
            n_iters += CHECK_INTERVAL
            if time.perf_counter() - wall_start >= duration:
                break
        profiler.disable()
        wall_elapsed = time.perf_counter() - wall_start
        rate = n_iters / wall_elapsed
        ns_per = 1e9 / rate
        time_str = f"{ns_per:.1f} ns/lookup" if ns_per < 1000 else f"{ns_per / 1000:.3f} µs/lookup"
        print(f"  Wall time: {wall_elapsed:.3f}s  "
              f"({rate:,.0f} lookups/s,  {time_str})")
        _print_profile_stats(profiler)

    # --- CompactTreeFlat ---
    _run_timed(
        "CompactTreeFlat.get_path()",
        lambda k0, k1, k2: flat_tree.get_path(k0, k1, k2),
        lambda k0, k1, k2: flat_tree.get_path(k0, k1, k2),
    )

    # --- CompactTree.get_path (comparison baseline) ---
    if compact_tree is not None:
        _run_timed(
            "CompactTree.get_path() [comparison baseline]",
            lambda k0, k1, k2: compact_tree.get_path(k0, k1, k2),
            lambda k0, k1, k2: compact_tree.get_path(k0, k1, k2),
        )

    # --- Plain Python dict (baseline) ---
    if use_dict:
        _run_timed(
            "dict[k0][k1][k2] [plain Python dict baseline]",
            lambda k0, k1, k2: d[k0][k1][k2],
            lambda k0, k1, k2: d[k0][k1][k2],
        )


# ---------------------------------------------------------------------------
# Serde profiling
# ---------------------------------------------------------------------------

def profile_serialize(tree: CompactTreeFlat) -> str:
    """Profile serialize() (uncompressed) for 10 seconds. Returns temp path."""
    with tempfile.NamedTemporaryFile(suffix=".ctflat", delete=False) as _f:
        tmp_path = _f.name

    tree.serialize(tmp_path)
    file_size = os.path.getsize(tmp_path)

    profiler = cProfile.Profile()
    CHECK_INTERVAL = 10
    DURATION = 10.0

    print(f"\nProfiling CompactTreeFlat.serialize() (uncompressed) for ~10s ...")
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
    """Profile CompactTreeFlat(path) (uncompressed) for 10 seconds."""
    _ = CompactTreeFlat(tmp_path)  # warmup

    profiler = cProfile.Profile()
    CHECK_INTERVAL = 10
    DURATION = 10.0

    print(f"\nProfiling CompactTreeFlat(path) (uncompressed) for ~10s ...")
    wall_start = time.perf_counter()
    n_iters = 0
    profiler.enable()
    while True:
        for _ in range(CHECK_INTERVAL):
            _ = CompactTreeFlat(tmp_path)
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
    parser = argparse.ArgumentParser(description="Profile CompactTreeFlat")
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
        "--compare",
        action="store_true",
        default=False,
        help=(
            "Also benchmark CompactTree.get_path() with identical inputs so "
            "the two implementations can be compared side-by-side. Only "
            "applies to --mode lookup and --mode both."
        ),
    )

    parser.add_argument(
        "--use-dict",
        action="store_true",
        default=False,
        dest="use_dict",
        help="Also benchmark plain dict[k0][k1][k2] as a baseline. Only applies to lookup/both modes.",
    )

    def _vocab_size_type(v: str) -> Optional[int]:
        if v.lower() == "none":
            return None
        try:
            value = int(v)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "vocab-size must be an integer >= 0 or 'None'"
            ) from exc
        if value < 0:
            raise argparse.ArgumentTypeError(
                "vocab-size must be 0, a positive integer, or 'None'"
            )
        return value

    parser.add_argument(
        "--vocab-size",
        type=_vocab_size_type,
        default=0,
        metavar="N",
        dest="vocab_size",
        help=(
            "vocabulary_size passed to from_dict (sets lru_cache maxsize on "
            "_val_trie). 0 (default) disables the LRU cache entirely. "
            "Pass 'None' to auto-size the cache to the full value vocabulary."
        ),
    )
    args = parser.parse_args()

    # Resolve L2 size
    if args.l2 is not None:
        l2_size = args.l2
    elif args.mode in ("serialize", "deserialize", "serde"):
        l2_size = 10_000
    else:
        l2_size = 173_000

    d = build_dict(l2_keys=l2_size)

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------
    if args.mode in ("build", "both"):
        flat_tree = profile_ingestion(d, vocabulary_size=args.vocab_size)
    else:
        print("\nBuilding CompactTreeFlat (unprofiled)...")
        t0 = time.perf_counter()
        flat_tree = CompactTreeFlat.from_dict(d, vocabulary_size=args.vocab_size)
        print(f"  Built in {time.perf_counter() - t0:.3f}s  "
              f"({len(flat_tree):,} leaf paths)")

    # ------------------------------------------------------------------
    # Comparison tree (built silently when --compare is requested)
    # ------------------------------------------------------------------
    compact_tree = None
    if args.compare and args.mode in ("lookup", "both"):
        from compact_tree import CompactTree
        print("\nBuilding CompactTree (unprofiled, for comparison)...")
        t0 = time.perf_counter()
        compact_tree = CompactTree.from_dict(d, vocabulary_size=args.vocab_size)
        print(f"  Built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Lookup phase
    # ------------------------------------------------------------------
    if args.mode in ("lookup", "both"):
        profile_lookup(
            flat_tree,
            compact_tree,
            d,
            duration=args.lookup_duration,
            use_dict=args.use_dict,
        )

    # ------------------------------------------------------------------
    # Serde phases
    # ------------------------------------------------------------------
    if args.mode in ("serialize", "serde"):
        tmp = profile_serialize(flat_tree)
        if args.mode == "serde":
            profile_deserialize(tmp)
        os.unlink(tmp)

    if args.mode == "deserialize":
        with tempfile.NamedTemporaryFile(suffix=".ctflat", delete=False) as _f:
            tmp = _f.name
        print("\nSerializing once (unprofiled) to temp file...")
        flat_tree.serialize(tmp)
        profile_deserialize(tmp)
        os.unlink(tmp)

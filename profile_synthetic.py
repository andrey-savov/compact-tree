"""Profile CompactTree ingestion of a synthetic 3-level nested dict.

Dict shape (unique keys per level):
    {0: 9, 1: 4, 2: 173_000}

Keys are drawn from N-gram permutations (N=1..7) from corpus.txt.
Values (leaves) are also drawn from the same permutation pool (N=1..7,
~4.6M entries), sampled with replacement so each leaf may differ.
"""

import cProfile
import io
import itertools
import pstats
import random
import time
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
L2_KEYS = 173_000
VALUE_POOL_CAP = 10_000   # cap unique leaf values to keep pre-warm tractable


def build_dict() -> dict:
    """Build the 3-level nested dict using corpus n-gram permutations."""
    corpus_path = Path(__file__).parent / "corpus.txt"

    print(f"Building vocabulary from corpus n-gram permutations (N=1..7)...")
    key_pool, value_pool = build_word_pools(corpus_path, max_n=7)

    need_keys = L0_KEYS + L1_KEYS + L2_KEYS
    if len(key_pool) < need_keys:
        raise ValueError(
            f"Not enough unique n-grams ({len(key_pool):,}) for "
            f"{need_keys:,} required keys"
        )

    l0_keys  = key_pool[:L0_KEYS]
    l1_keys  = key_pool[L0_KEYS : L0_KEYS + L1_KEYS]
    l2_keys  = key_pool[L0_KEYS + L1_KEYS : L0_KEYS + L1_KEYS + L2_KEYS]

    # Cap the value pool so the number of unique leaf values stays tractable
    value_pool = value_pool[:VALUE_POOL_CAP]

    print(f"  L0 keys: {len(l0_keys)}, L1 keys: {len(l1_keys)}, "
          f"L2 keys: {len(l2_keys)}, value pool: {len(value_pool):,}")

    rng = random.Random(99)
    t1 = time.perf_counter()
    print("Building nested dict...")
    d: dict = {}
    for k0 in l0_keys:
        d[k0] = {}
        for k1 in l1_keys:
            inner = {k2: rng.choice(value_pool) for k2 in l2_keys}
            d[k0][k1] = inner

    total_leaves = L0_KEYS * L1_KEYS * L2_KEYS
    print(f"  Dict built in {time.perf_counter() - t1:.3f}s  "
          f"({total_leaves:,} leaf entries)")
    return d


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_ingestion(d: dict) -> CompactTree:
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
    vocab_size = len(all_keys) + len(all_values)
    print(f"  Unique keys: {len(all_keys):,}, unique values: {len(all_values):,} "
          f"-> vocabulary_size=None (unbounded)")

    profiler = cProfile.Profile()

    print("\nProfiling CompactTree.from_dict() ...")
    wall_start = time.perf_counter()
    profiler.enable()
    tree = CompactTree.from_dict(d)
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = build_dict()
    tree = profile_ingestion(d)

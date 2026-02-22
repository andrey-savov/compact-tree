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
  python profile_synthetic.py --mode lookup --vocab-size None  # enable auto-sized LRU cache
  python profile_synthetic.py --mode lookup --use-parquet          # PyArrow table filter-based lookup
  python profile_synthetic.py --mode lookup --use-parquet --parquet-sorted  # PyArrow sorted + bisect
  python profile_synthetic.py --mode build  --use-parquet          # profile PyArrow table construction
  python profile_synthetic.py --mode both   --use-parquet          # profile table build + lookup
  python profile_synthetic.py --mode lookup --use-parquet-map      # PyArrow nested map column lookup
  python profile_synthetic.py --mode build  --use-parquet-map      # profile nested map table construction
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
from typing import TYPE_CHECKING, Any, Generator, Optional
from pathlib import Path

if TYPE_CHECKING:
    import pyarrow as pa

from compact_tree import CompactTree

# ---------------------------------------------------------------------------
# N-gram permutation generator
# ---------------------------------------------------------------------------

def _ngram_permutations(words: list[str], max_n: int = 7) -> Generator[str, None, None]:
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


def build_dict(l2_keys: int = 173_000) -> dict[str, dict[str, dict[str, str]]]:
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
# PyArrow table helpers
# ---------------------------------------------------------------------------

def dict_to_arrow_table(d: dict[str, dict[str, dict[str, str]]], sorted_table: bool = False) -> pa.Table:
    """Flatten the 3-level dict to a PyArrow table with 4 string columns.

    Columns: ``l0_key``, ``l1_key``, ``l2_key``, ``value``.

    When *sorted_table* is True, a combined ``_skey`` column
    (``l0_key + '\\x00' + l1_key + '\\x00' + l2_key``) is appended and the
    table is sorted by it ascending so that :func:`parquet_sorted_lookup` can
    use :mod:`bisect` for O(log N) lookups.
    """
    import pyarrow as pa  # lazy import — only required when --use-parquet is active

    l0_col: list[str] = []
    l1_col: list[str] = []
    l2_col: list[str] = []
    val_col: list[str] = []

    for k0, sub1 in d.items():
        for k1, sub2 in sub1.items():
            for k2, v in sub2.items():
                l0_col.append(k0)
                l1_col.append(k1)
                l2_col.append(k2)
                val_col.append(v)

    schema = pa.schema([
        ("l0_key", pa.string()),
        ("l1_key", pa.string()),
        ("l2_key", pa.string()),
        ("value",  pa.string()),
    ])
    table = pa.table(
        {
            "l0_key": pa.array(l0_col, type=pa.string()),
            "l1_key": pa.array(l1_col, type=pa.string()),
            "l2_key": pa.array(l2_col, type=pa.string()),
            "value":  pa.array(val_col, type=pa.string()),
        },
        schema=schema,
    )

    if sorted_table:
        import pyarrow.compute as pc
        skey_col = [
            k0 + "\x00" + k1 + "\x00" + k2
            for k0, k1, k2 in zip(l0_col, l1_col, l2_col)
        ]
        table = table.append_column(
            "_skey", pa.array(skey_col, type=pa.string())
        )
        sort_idx = pc.sort_indices(table, sort_keys=[("_skey", "ascending")])  # type: ignore[attr-defined]
        table = table.take(sort_idx)

    return table


def parquet_filter_lookup(table: pa.Table, k0: str, k1: str, k2: str) -> str:
    """Point lookup on an unsorted PyArrow table using ``pc.equal`` + ``table.filter``.

    Raises :class:`KeyError` when no matching row is found.
    """
    import pyarrow.compute as pc

    mask = pc.and_(  # type: ignore[attr-defined]
        pc.and_(  # type: ignore[attr-defined]
            pc.equal(table.column("l0_key"), k0),  # type: ignore[attr-defined]
            pc.equal(table.column("l1_key"), k1),  # type: ignore[attr-defined]
        ),
        pc.equal(table.column("l2_key"), k2),  # type: ignore[attr-defined]
    )
    result = table.filter(mask)
    if result.num_rows == 0:
        raise KeyError((k0, k1, k2))
    return result.column("value")[0].as_py()


def parquet_sorted_lookup(
    table: pa.Table,
    skey_list: list[str],
    k0: str,
    k1: str,
    k2: str,
) -> str:
    """O(log N) point lookup on a pre-sorted PyArrow table.

    *skey_list* must be the ``_skey`` column materialised as a Python list
    (call ``table.column("_skey").to_pylist()`` once before the benchmark loop
    and reuse it).  :mod:`bisect` locates the row index; the value is then
    fetched directly from the PyArrow ``value`` column.

    Raises :class:`KeyError` when no matching row is found.
    """
    import bisect

    target = k0 + "\x00" + k1 + "\x00" + k2
    idx = bisect.bisect_left(skey_list, target)
    if idx >= len(skey_list) or skey_list[idx] != target:
        raise KeyError((k0, k1, k2))
    return table.column("value")[idx].as_py()


def profile_parquet_build(d: dict[str, dict[str, dict[str, str]]], parquet_sorted: bool = False) -> pa.Table:
    """Profile :func:`dict_to_arrow_table` and print a cProfile summary.

    Returns the built table so the caller can pass it directly to
    :func:`profile_lookup` when running ``--mode both --use-parquet``.
    """
    mode_label = "sorted (_skey column + sort)" if parquet_sorted else "unsorted"
    print(f"\nProfiling dict_to_arrow_table() [{mode_label}] ...")

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    table = dict_to_arrow_table(d, sorted_table=parquet_sorted)
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    print(f"  Wall time: {wall_elapsed:.3f}s")
    print(f"  Rows: {table.num_rows:,}  |  Size: {table.nbytes / 1_048_576:.1f} MiB")

    _print_profile_stats(profiler)
    return table


def dict_to_arrow_map_table(d: dict[str, dict[str, dict[str, str]]]) -> pa.Table:
    """Build a single-cell PyArrow table using a triply-nested map column.

    The table has **exactly 1 row** and **1 column**:

    * ``data`` — ``map<string, map<string, map<string, string>>>``
                 the entire 3-level dict stored in one PyArrow map scalar.

    Lookup is performed by chaining ``pc.map_lookup`` + ``pc.list_flatten``
    three times, keeping all data in Arrow memory until the final string
    value is extracted with a single ``.as_py()`` call.
    """
    import pyarrow as pa  # lazy import

    l2_type  = pa.map_(pa.string(), pa.string())
    l1_type  = pa.map_(pa.string(), l2_type)
    l0_type  = pa.map_(pa.string(), l1_type)

    # Build the single cell value: a list of (k0, [(k1, [(k2, v) ...]) ...])
    l0_pairs: list[tuple] = []
    for k0, sub1 in d.items():
        l1_pairs: list[tuple] = []
        for k1, sub2 in sub1.items():
            l1_pairs.append((k1, list(sub2.items())))
        l0_pairs.append((k0, l1_pairs))

    return pa.table(
        {"data": pa.array([l0_pairs], type=l0_type)},
        schema=pa.schema([("data", l0_type)]),
    )


def parquet_map_lookup(table: pa.Table, k0: str, k1: str, k2: str) -> str:
    """Point lookup on a single-cell triply-nested-map PyArrow table.

    All three levels are traversed in Arrow-land using chained
    ``pc.map_lookup`` + ``pc.list_flatten`` calls (both C++).  Only the
    final matched string scalar is materialised via ``.as_py()``.

    Chain::

        col  : MapArray<str, map<str, map<str, str>>>   (1 element)
         → map_lookup(k0) → ListArray<map<str, map<str,str>>>  (1 element)
         → list_flatten   → MapArray<str, map<str,str>>        (0 or 1)
         → map_lookup(k1) → ListArray<map<str,str>>            (0 or 1)
         → list_flatten   → MapArray<str,str>                  (0 or 1)
         → map_lookup(k2) → ListArray<str>                     (0 or 1)
         → list_flatten   → StringArray                        (0 or 1)

    Raises :class:`KeyError` on any miss.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    col = table.column("data")  # 1-element MapArray

    l1_list = pc.map_lookup(col, pa.scalar(k0, pa.string()), "all")  # type: ignore[attr-defined]
    l1_flat = pc.list_flatten(l1_list)  # type: ignore[attr-defined]  # MapArray<str,map<str,str>>
    if len(l1_flat) == 0:
        raise KeyError((k0, k1, k2))

    l2_list = pc.map_lookup(l1_flat, pa.scalar(k1, pa.string()), "all")  # type: ignore[attr-defined]
    l2_flat = pc.list_flatten(l2_list)  # type: ignore[attr-defined]  # MapArray<str,str>
    if len(l2_flat) == 0:
        raise KeyError((k0, k1, k2))

    val_list = pc.map_lookup(l2_flat, pa.scalar(k2, pa.string()), "all")  # type: ignore[attr-defined]
    val_flat = pc.list_flatten(val_list)  # type: ignore[attr-defined]  # StringArray
    if len(val_flat) == 0:
        raise KeyError((k0, k1, k2))

    return val_flat[0].as_py()


def profile_parquet_map_build(d: dict[str, dict[str, dict[str, str]]]) -> pa.Table:
    """Profile :func:`dict_to_arrow_map_table` and print a cProfile summary.

    Returns the built table so the caller can reuse it in subsequent lookup
    profiling without rebuilding (``--mode both --use-parquet-map``).
    """
    print("\nProfiling dict_to_arrow_map_table() [1-row map<str,map<str,map<str,str>>>] ...")

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    table = dict_to_arrow_map_table(d)
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    print(f"  Wall time: {wall_elapsed:.3f}s")
    print(f"  Rows: {table.num_rows:,}  |  Size: {table.nbytes / 1_048_576:.1f} MiB")

    _print_profile_stats(profiler)
    return table


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_ingestion(d: dict[str, dict[str, dict[str, str]]], vocabulary_size: Optional[int] = 0) -> CompactTree:
    """Profile CompactTree.from_dict(d) and print a summary."""
    # Estimate unique keys and values to size the LRU cache exactly.
    all_keys: set[str] = set()
    all_values: set[str] = set()
    def _walk(node: dict[str, Any]) -> None:
        for k, v in node.items():
            all_keys.add(k)
            if isinstance(v, dict):
                _walk(v)
            else:
                all_values.add(str(v))
    _walk(d)
    vocab_hint = (
        f"vocabulary_size={vocabulary_size} (cache DISABLED — every lookup calls _index_uncached, default)"
        if vocabulary_size == 0
        else (
            "vocabulary_size=None (auto-sized per trie: "
            f"keys={len(all_keys):,}, values={len(all_values):,})"
        )
        if vocabulary_size is None
        else f"vocabulary_size={vocabulary_size!r} (capped cache)"
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

def profile_lookup(
    tree: Optional[CompactTree],
    d: dict[str, dict[str, dict[str, str]]],
    duration: float = 10.0,
    miss_ratio: float = 0.1,
    use_get_path: bool = False,
    use_dict: bool = False,
    use_parquet: bool = False,
    parquet_sorted: bool = False,
    parquet_table: Optional[pa.Table] = None,
    use_parquet_map: bool = False,
    parquet_map_table: Optional[pa.Table] = None,
) -> None:
    """Profile random leaf lookups for approximately *duration* seconds.

    Key generation is O(1) per lookup: precompute the small key lists for
    each level, then use ``rng.randrange(len(keys))`` + direct list indexing
    to pick a key.  No path list is precomputed.

    ``miss_ratio`` fraction of lookups intentionally use a key from the wrong
    level (guaranteed miss) to exercise the KeyError / __contains__ path.

    Target selection (mutually exclusive):

    * default — ``tree[k0][k1][k2]`` against the CompactTree
    * ``use_get_path`` — ``tree.get_path(k0, k1, k2)``
    * ``use_dict`` — ``d[k0][k1][k2]`` against the plain Python dict
    * ``use_parquet`` — flat 4-column PyArrow table (filter or sorted bisect)
    * ``use_parquet_map`` — 1-row map<str,map<str,map<str,str>>> column;
                            3× ``pc.map_lookup`` + ``pc.list_flatten`` (all C++)

    When *use_parquet* / *use_parquet_map* is True and the corresponding table
    is None, the table is built from *d* here (outside the timed loop).  Pass
    a pre-built table (e.g. from :func:`profile_parquet_build`) to avoid
    building it twice in ``--mode both``.
    """
    # Extract key lists once — tiny: 9, 4, 173K entries.
    l0_keys = list(d.keys())
    # All L0 children have the same L1 keys; sample from the first.
    l1_keys = list(next(iter(d.values())).keys())
    # All L1 children share the same L2 key set; sample from d[l0][l1].
    l2_keys = list(next(iter(next(iter(d.values())).values())).keys())
    n0, n1, n2 = len(l0_keys), len(l1_keys), len(l2_keys)
    print(f"\nKey lists: L0={n0}, L1={n1}, L2={n2:,}")

    # ------------------------------------------------------------------
    # Per-target setup and warmup
    # ------------------------------------------------------------------
    skey_list: Optional[list[str]] = None  # only used by parquet_sorted path

    if use_parquet_map:
        if parquet_map_table is None:
            print("  Building PyArrow nested-map table from dict (unprofiled)...")
            t0 = time.perf_counter()
            parquet_map_table = dict_to_arrow_map_table(d)
            print(f"  Table built in {time.perf_counter() - t0:.3f}s")
        assert parquet_map_table is not None
        print(f"  Table: {parquet_map_table.num_rows:,} rows, "
              f"{parquet_map_table.nbytes / 1_048_576:.1f} MiB")
        print("  Using PyArrow nested-map (3× pc.map_lookup + pc.list_flatten, 1 .as_py())")
    elif use_parquet:
        if parquet_table is None:
            print("  Building PyArrow table from dict (unprofiled)...")
            t0 = time.perf_counter()
            parquet_table = dict_to_arrow_table(d, sorted_table=parquet_sorted)
            print(f"  Table built in {time.perf_counter() - t0:.3f}s")
        assert parquet_table is not None
        print(f"  Table: {parquet_table.num_rows:,} rows, "
              f"{parquet_table.nbytes / 1_048_576:.1f} MiB")
        if parquet_sorted:
            print("  Extracting _skey list for bisect (one-time)...")
            skey_list = parquet_table.column("_skey").to_pylist()
            print("  Using PyArrow table (sorted, O(log N) bisect + column access)")
        else:
            print("  Using PyArrow table (filter: pc.equal + table.filter)")
    elif use_dict:
        print("  Using plain Python dict — no warmup needed")
    else:
        cache_disabled = getattr(tree, '_key_vocab_size', None) == 0
        if cache_disabled:
            print("  Cache disabled (vocab_size=0) — skipping warmup, profiling _index_uncached directly")
        else:
            for _ in range(2_000):
                _ = tree[l0_keys[_ % n0]]  # type: ignore[index]
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
                    if use_parquet_map:
                        _ = parquet_map_lookup(parquet_map_table, k0, k1, k2)  # type: ignore[arg-type]
                    elif use_parquet:
                        if parquet_sorted:
                            assert skey_list is not None
                            _ = parquet_sorted_lookup(parquet_table, skey_list, k0, k1, k2)
                        else:
                            _ = parquet_filter_lookup(parquet_table, k0, k1, k2)
                    elif use_dict:
                        _ = d[k0][k1][k2]
                    elif use_get_path:
                        _ = tree.get_path(k0, k1, k2)  # type: ignore[union-attr]
                    else:
                        _ = tree[k0][k1][k2]  # type: ignore[index]
                except KeyError:
                    pass
            else:
                if use_parquet_map:
                    _ = parquet_map_lookup(parquet_map_table, k0, k1, k2)  # type: ignore[arg-type]
                elif use_parquet:
                    if parquet_sorted:
                        assert skey_list is not None
                        _ = parquet_sorted_lookup(parquet_table, skey_list, k0, k1, k2)
                    else:
                        _ = parquet_filter_lookup(parquet_table, k0, k1, k2)
                elif use_dict:
                    _ = d[k0][k1][k2]
                elif use_get_path:
                    _ = tree.get_path(k0, k1, k2)  # type: ignore[union-attr]
                else:
                    _ = tree[k0][k1][k2]  # type: ignore[index]
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

    # --use-dict and --use-parquet are mutually exclusive lookup-target overrides.
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--use-dict",
        action="store_true",
        default=False,
        dest="use_dict",
        help="Benchmark the plain Python dict instead of CompactTree (baseline comparison)",
    )
    target_group.add_argument(
        "--use-parquet",
        action="store_true",
        default=False,
        dest="use_parquet",
        help=(
            "Benchmark a PyArrow table instead of CompactTree. "
            "Supported with --mode build, lookup, and both. "
            "Uses filter-based lookup (pc.equal + table.filter) by default; "
            "add --parquet-sorted for O(log N) bisect on a pre-sorted table."
        ),
    )
    target_group.add_argument(
        "--use-parquet-map",
        action="store_true",
        default=False,
        dest="use_parquet_map",
        help=(
            "Benchmark a compact PyArrow table with a triply nested map column "
            "(map<string, map<string, map<string, string>>>). "
            "Lookup uses pc.map_lookup + pc.list_flatten at all three levels "
            "(no Python-dict lookup). "
            "Supported with --mode build, lookup, and both."
        ),
    )

    parser.add_argument(
        "--parquet-sorted",
        action="store_true",
        default=False,
        dest="parquet_sorted",
        help=(
            "With --use-parquet: sort the table by (l0_key, l1_key, l2_key) and "
            "use bisect for O(log N) lookups instead of a full-table filter scan."
        ),
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
        help="vocabulary_size passed to from_dict (sets lru_cache maxsize). "
             "0 (default) disables the LRU cache entirely. "
             "Pass 'None' to auto-size the cache to the full vocabulary.",
    )
    args = parser.parse_args()

    # Post-parse validation
    if args.parquet_sorted and not args.use_parquet:
        parser.error("--parquet-sorted requires --use-parquet")
    if args.use_parquet and args.mode in ("serialize", "deserialize", "serde"):
        parser.error("--use-parquet is not supported with serde modes (serialize/deserialize/serde)")
    if args.use_parquet_map and args.mode in ("serialize", "deserialize", "serde"):
        parser.error("--use-parquet-map is not supported with serde modes (serialize/deserialize/serde)")
    if args.use_get_path and args.use_dict:
        parser.error("--use-get-path cannot be combined with --use-dict; get_path is only supported for CompactTree lookups")
    if args.use_get_path and (args.use_parquet or args.use_parquet_map):
        parser.error("--use-get-path is only supported when benchmarking CompactTree (not with --use-parquet or --use-parquet-map)")

    # Resolve L2 size: explicit > mode default > global default
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
    parquet_table: Optional[pa.Table] = None       # set when --use-parquet is active
    parquet_map_table: Optional[pa.Table] = None   # set when --use-parquet-map is active

    if args.use_parquet_map:
        if args.mode in ("build", "both"):
            parquet_map_table = profile_parquet_map_build(d)
        else:
            # Build PyArrow map table silently for lookup/other modes.
            print("\nBuilding PyArrow map table (unprofiled)...")
            t0 = time.perf_counter()
            parquet_map_table = dict_to_arrow_map_table(d)
            print(f"  Built in {time.perf_counter() - t0:.3f}s")
        tree = None  # CompactTree not needed when benchmarking PyArrow map
    elif args.use_parquet:
        if args.mode in ("build", "both"):
            parquet_table = profile_parquet_build(d, parquet_sorted=args.parquet_sorted)
        else:
            # Build PyArrow table silently for lookup/other modes.
            print("\nBuilding PyArrow table (unprofiled)...")
            t0 = time.perf_counter()
            parquet_table = dict_to_arrow_table(d, sorted_table=args.parquet_sorted)
            print(f"  Built in {time.perf_counter() - t0:.3f}s")
        tree = None  # CompactTree not needed when benchmarking PyArrow
    else:
        if args.mode in ("build", "both"):
            tree = profile_ingestion(d, vocabulary_size=args.vocab_size)
        else:
            # Build silently for all other modes.
            print("\nBuilding CompactTree (unprofiled)...")
            t0 = time.perf_counter()
            tree = CompactTree.from_dict(d, vocabulary_size=args.vocab_size)
            print(f"  Built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Lookup phase
    # ------------------------------------------------------------------
    if args.mode in ("lookup", "both"):
        profile_lookup(
            tree, d,
            duration=args.lookup_duration,
            use_get_path=args.use_get_path,
            use_dict=args.use_dict,
            use_parquet=args.use_parquet,
            parquet_sorted=args.parquet_sorted,
            parquet_table=parquet_table,
            use_parquet_map=args.use_parquet_map,
            parquet_map_table=parquet_map_table,
        )

    # ------------------------------------------------------------------
    # Serde phases (CompactTree only)
    # ------------------------------------------------------------------
    if args.mode in ("serialize", "serde"):
        assert tree is not None
        tmp = profile_serialize(tree)
        if args.mode == "serde":
            profile_deserialize(tmp)
        os.unlink(tmp)

    if args.mode == "deserialize":
        assert tree is not None
        with tempfile.NamedTemporaryFile(suffix=".ctree", delete=False) as _f:
            tmp = _f.name
        print("\nSerializing once (unprofiled) to temp file...")
        tree.serialize(tmp)
        profile_deserialize(tmp)
        os.unlink(tmp)


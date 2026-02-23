# Performance Benchmarks

Lookup throughput comparison across CompactTree, plain Python `dict`, and
three PyArrow table layouts.

---

## Platforms

### Platform A — Windows 11

| Field | Value |
|---|---|
| Date | 2026-02-22 |
| OS | Windows 11 (10.0.26200) |
| CPU | Intel Core Ultra 9 285H |
| Python | 3.14.3 (CPython, MSC v.1944 64-bit, AMD64) |
| PyArrow | 23.0.1 |

### Platform B — Linux (WSL2)

| Field | Value |
|---|---|
| Date | 2026-02-22 |
| OS | Linux (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2) |
| CPU | Intel Core Ultra 9 285H |
| Python | 3.14.3 (CPython, GCC 13.3.0, 64-bit, x86_64) |
| PyArrow | 23.0.1 |

---

## Dataset

| Parameter | Value |
|---|---|
| Shape | 9 L0 keys × 4 L1 keys × 173,000 L2 keys |
| Total leaf entries | 6,228,000 |
| Leaf value pool | 10,000 unique strings (n-gram permutations) |
| Key strings | N-gram permutations from `corpus.txt` (N=1..7) |

---

## Lookup benchmark methodology

* Random leaf lookups driven by `random.Random(42)`.
* **10 % miss ratio**: on miss iterations the L2 key is swapped for an L0 key
  (guaranteed `KeyError`) to exercise the miss path.
* The timed loop runs for **5 seconds** wall-clock per target; iteration count
  divided by elapsed time gives the final rate.
* Build time is excluded from lookup timing in all cases.
* `cProfile` is active during the timed loop (adds a small, consistent overhead
  to all targets, so relative numbers remain comparable).

---

## Results — lookup throughput

### Platform A — Windows 11

| Target | Lookups / s | µs / lookup | vs dict |
|---|---|---|---|
| `dict[k0][k1][k2]` | 235,958 | 4.2 | 1.0× (baseline) |
| `CompactTreeFlat.get_path()` † | 194,220 | 5.1 | 0.85× †|
| `CompactTree.get_path()` (C extension) † | 141,224 | 7.1 | 0.62× †|
| `CompactTree[k0][k1][k2]` (no LRU cache, default) | 124,384 | 8.0 | 0.53× |
| `CompactTree[k0][k1][k2]` (no LRU cache, `shared_trie=True`) | 123,573 | 8.1 | 0.52× |
| `CompactTree[k0][k1][k2]` (LRU cache, `vocabulary_size=None`) | 116,589 | 8.6 | 0.49× |
| PyArrow flat table, sorted + bisect | 113,450 | 8.8 | 0.48× |
| PyArrow nested map (`pc.map_lookup` × 3) | 64 | 15,734 | 0.00027× |
| PyArrow flat table, filter scan | 34 | 29,302 | 0.00014× |

### Platform B — Linux (WSL2)

| Target | Lookups / s | µs / lookup | vs dict |
|---|---|---|---|
| `dict[k0][k1][k2]` | 245,520 | 4.1 | 1.0× (baseline) |
| `CompactTreeFlat.get_path()` † | 190,870 | 5.2 | 0.82× †|
| `CompactTree.get_path()` (C extension) † | 157,793 | 6.3 | 0.67× †|
| `CompactTree[k0][k1][k2]` (no LRU cache, default) | 133,796 | 7.5 | 0.55× |
| `CompactTree[k0][k1][k2]` (no LRU cache, `shared_trie=True`) | 131,422 | 7.6 | 0.54× |
| `CompactTree[k0][k1][k2]` (LRU cache, `vocabulary_size=None`) | 133,682 | 7.5 | 0.54× |
| PyArrow flat table, sorted + bisect | 120,420 | 8.3 | 0.49× |
| PyArrow nested map (`pc.map_lookup` × 3) | 67 | 14,857 | 0.00027× |
| PyArrow flat table, filter scan | 41 | 24,609 | 0.00017× |

### Notes

* **CompactTree vs sorted-bisect PyArrow** are within measurement noise of
  each other (~7.1–8.8 µs across platforms). Both are roughly 2× slower than
  a native dict on both platforms.
* **Linux vs Windows**: Linux (WSL2) shows ~4–8% higher throughput for dict and
  CompactTree (~246K vs ~236K and ~134K vs ~124K lookups/s), consistent with
  lower syscall/scheduler overhead in native Linux execution.
* **LRU cache vs no cache** (now default): with `vocabulary_size=0` (the
  default) the per-instance `lru_cache` on `index()` is disabled and every
  call goes directly to `_index_uncached` (pure Python trie traversal or C
  extension lookup). Because the LRU cache introduces its own hashing and
  bookkeeping overhead that largely cancels out the benefit of skipping the
  trie walk on a hit, the two variants measure within noise of each other on
  both platforms (~7–9 µs). The cache is therefore disabled by default; pass
  `vocabulary_size=None` to `from_dict()` to re-enable auto-sized caching.
* **`shared_trie=True`** uses a single `MarisaTrie` for both keys and values
  instead of two separate tries. Lookup throughput is identical within
  measurement noise (8.0 vs 8.1 µs on Windows) because the dominant cost is
  the C-level `TreeIndex.get` traversal, which is unchanged. The benefit is
  lower memory usage when key and value vocabularies overlap significantly;
  pass `shared_trie=True` to `from_dict()` to enable it.
* **`CompactTreeFlat.get_path()`** (†) is measured via `profile_compact_tree_flat.py`
  using its own timed loop (10 s, same L2=173,000 dataset, 10% miss ratio).
  The vs-dict ratios for the `†` rows use the dict baseline from that same
  run — Win: 228,998 /s (4.4 µs), Linux: 233,879 /s (4.3 µs) — rather than
  the separate `profile_synthetic.py` dict figures above.
  `CompactTreeFlat` is ~38% faster than `CompactTree.get_path()` on Windows
  and ~21% faster on Linux, because a full-path lookup reduces to a single
  Python `dict.__getitem__` call plus one `MarisaTrie.restore_key()` call,
  with no CSR trie traversal.
* **PyArrow flat filter** and **nested map** are O(N) scans over all 173K rows
  per lookup — roughly 1,500–3,500× slower than the indexed approaches. They
  are included for completeness, not as practical lookup strategies.
* The sorted-bisect approach materialises the `_skey` column as a Python
  `list[str]` once before the benchmark loop; `bisect.bisect_left` then
  provides O(log N) row location followed by a single Arrow column access.

---

## Build throughput

Build profiling uses `cProfile` for a single construction pass (not a timed
loop). Representative wall times at L2 = 173,000:

| Target | Wall time (s) | Output size |
|---|---|---|
| `CompactTree.from_dict()` (separate tries, default) | ~11 s | — |
| `CompactTree.from_dict()` (`shared_trie=True`) | ~11 s | — |
| `dict_to_arrow_table()` (unsorted) | ~0.5 s | ~50 MiB |
| `dict_to_arrow_table()` (sorted, +`_skey`) | ~0.7 s | ~65 MiB |
| `dict_to_arrow_map_table()` (1-row nested map) | ~3–4 s | ~45 MiB |

---

## How to reproduce

```bash
# Lookup benchmarks (10 s each)
python profile_compact_tree_flat.py --mode both --compare --use-dict --lookup-duration 10          # CompactTreeFlat vs CompactTree.get_path() vs dict
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10                         # no LRU cache (default)
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --shared-trie           # shared trie, no LRU cache
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --vocab-size None       # LRU cache, auto-sized
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --use-dict
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --use-parquet --parquet-sorted
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --use-parquet-map
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 10 --use-parquet

# Build benchmarks
python profile_synthetic.py --mode build --l2 173000
python profile_synthetic.py --mode build --l2 173000 --shared-trie
python profile_synthetic.py --mode build --l2 173000 --use-parquet
python profile_synthetic.py --mode build --l2 173000 --use-parquet --parquet-sorted
python profile_synthetic.py --mode build --l2 173000 --use-parquet-map
```

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
| `CompactTree[k0][k1][k2]` (LRU cache, default) | 116,589 | 8.6 | 0.49× |
| `CompactTree[k0][k1][k2]` (no LRU cache, `--vocab-size 0`) | 119,100 | 8.4 | 0.50× |
| PyArrow flat table, sorted + bisect | 113,450 | 8.8 | 0.48× |
| PyArrow nested map (`pc.map_lookup` × 3) | 64 | 15,734 | 0.00027× |
| PyArrow flat table, filter scan | 34 | 29,302 | 0.00014× |

### Platform B — Linux (WSL2)

| Target | Lookups / s | µs / lookup | vs dict |
|---|---|---|---|
| `dict[k0][k1][k2]` | 256,157 | 3.9 | 1.0× (baseline) |
| `CompactTree[k0][k1][k2]` (LRU cache, default) | 141,112 | 7.1 | 0.55× |
| `CompactTree[k0][k1][k2]` (no LRU cache, `--vocab-size 0`) | 137,149 | 7.3 | 0.54× |
| PyArrow flat table, sorted + bisect | 122,284 | 8.2 | 0.48× |
| PyArrow nested map (`pc.map_lookup` × 3) | 72 | 13,981 | 0.00028× |
| PyArrow flat table, filter scan | 42 | 23,917 | 0.00016× |

### Notes

* **CompactTree vs sorted-bisect PyArrow** are within measurement noise of
  each other (~7.1–8.8 µs across platforms). Both are roughly 2× slower than
  a native dict on both platforms.
* **Linux vs Windows**: Linux (WSL2) shows ~20% higher throughput for dict and
  CompactTree (~256K vs ~236K and ~141K vs ~117K lookups/s), consistent with
  lower syscall/scheduler overhead in native Linux execution.
* **LRU cache vs no cache**: with `--vocab-size 0` the per-instance
  `lru_cache` on `index()` is disabled and every call goes directly to
  `_index_uncached` (pure Python trie traversal or C extension lookup without
  the cache layer). At L2 = 173K, the default cache is sized to hold the full
  vocabulary (~173K keys) so the cache fill-ratio is near 100% after warmup —
  yet the two variants measure within noise of each other on both platforms.
  The LRU cache introduces its own hashing and bookkeeping overhead that
  largely cancels out the benefit of skipping the trie walk on a hit, making
  the cache neutral at this key-count and access pattern.
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
| `CompactTree.from_dict()` | ~6–7 s | — |
| `dict_to_arrow_table()` (unsorted) | ~0.5 s | ~50 MiB |
| `dict_to_arrow_table()` (sorted, +`_skey`) | ~0.7 s | ~65 MiB |
| `dict_to_arrow_map_table()` (1-row nested map) | ~3–4 s | ~45 MiB |

---

## How to reproduce

```bash
# Lookup benchmarks (5 s each)
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5 --vocab-size 0
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5 --use-dict
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5 --use-parquet --parquet-sorted
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5 --use-parquet-map
python profile_synthetic.py --mode lookup --l2 173000 --lookup-duration 5 --use-parquet

# Build benchmarks
python profile_synthetic.py --mode build --l2 173000
python profile_synthetic.py --mode build --l2 173000 --use-parquet
python profile_synthetic.py --mode build --l2 173000 --use-parquet --parquet-sorted
python profile_synthetic.py --mode build --l2 173000 --use-parquet-map
```

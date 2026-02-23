# CompactTree Benchmark Results

## Latest Benchmark Results — v3.0.0 (2026-02-22)

**Machine**: Windows 11, Intel Core Ultra 9 285H  
**Python**: 3.14.3 (CPython, MSC v.1944 64-bit)  
**C extension**: `_marisa_ext` compiled (`TrieIndex` + `TreeIndex`)  
**Date**: 2026-02-22

```
pytest test_compact_tree.py::TestLoadPerformance --benchmark-only -v
```

| Test | Min | Max | Mean | StdDev | Rounds | Unit |
|------|-----|-----|------|--------|--------|------|
| **test_build_compact_tree_from_cooccurrence** | **15.48** | **24.58** | **17.31** | 1.65 | 57 | **ms** |
| test_tree_lookups_at_different_depths | 0.88 | 53.68 | 1.02 | 0.49 | 192,308 | μs |
| test_serialization_performance | 343.30 | 1,157.50 | 426.84 | 136.63 | 80 | μs |
| test_deserialization_performance | 1,739.80 | 7,027.80 | 2,204.95 | 855.77 | 37 | μs |

### Performance Summary

**Build time** (37K-entry co-occurrence dict): **17.31 ms mean** (57.77 builds/s)  
**Lookup throughput**: **984,873 /s** (1.02 µs/lookup) — C extension active  
**Serialization**: **2,343 /s** (426.84 µs/op)  
**Deserialization**: **454 /s** (2,204.95 µs/op)

**Key Metrics:**
- **37,019 total entries** in co-occurrence dictionary
- **~985K lookups/s** with C extension (`TreeIndex`)
- ~17ms build time (down from ~29ms in v2.0.0 baseline due to C extension navigation)

---

## Historical Benchmark Results

### v2.1.0 (2026-02-21) — Python 3.13, pure-Python fallback (no C extension)

| Test | Min | Max | Mean | StdDev | Rounds | Unit |
|------|-----|-----|------|--------|--------|------|
| build_compact_tree_from_cooccurrence | 26.22 | 33.99 | 28.89 | 1.31 | 31 | ms |
| tree_lookups_at_different_depths | 4.60 | 197.70 | 5.45 | 2.68 | 136,987 | μs |
| serialization_performance | 258.70 | 6,691.90 | 332.35 | 196.83 | 1,155 | μs |
| deserialization_performance | 1,298.10 | 2,548.70 | 1,493.41 | 281.69 | 45 | μs |

### Pre-v1.2.0 Optimization History

#### Optimization #4: LRU Cache (2026-02-15)
- **Speedup:** 2,400ms → 29ms (84x improvement)
- **Implementation:** Instance-level OrderedDict cache with 4,096 entry limit
- **Impact:** Eliminates redundant trie traversals for repeated keys

### Optimization #3: O(1) Label Access (2026-02-15)
- **Speedup:** Pre-computed label offsets (8x faster label lookups)
- **Impact:** `_get_label()` from O(n) to O(1)

### Optimization #2: Key Lookup Caching (2026-02-15)
- **Speedup:** Cache key indices before sorting
- **Impact:** 34% reduction in MarisaTrie lookups

### Optimization #1: Optimized Count Computation (2026-02-15)
- **Speedup:** 14x faster MarisaTrie building
- **Impact:** Use children_map instead of LOUDS navigation

## Historical Benchmark Results

#### Before Optimizations (v1.1.0 baseline, 2026-02-15)

| Test | Min (ms) | Max (ms) | Mean (ms) | StdDev | Rounds |
|------|----------|----------|-----------|--------|--------|
| build_compact_tree_from_cooccurrence | 5,300.26 | 5,393.32 | 5,340.81 | 34.31 | 5 |
| tree_lookups_at_different_depths | 0.52 | 1.53 | 0.58 | 0.09 | 1,794 |
| serialization_performance | 0.29 | 0.74 | 0.37 | 0.08 | 67 |
| deserialization_performance | 1.30 | 2.74 | 1.51 | 0.31 | 36 |

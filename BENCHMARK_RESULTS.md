# CompactTree Benchmark Results

**Machine**: ultra  
**Python**: 3.13.12  
**Date**: 2026-02-15

## Latest Benchmark Results (After All Optimizations)

| Test | Min | Max | Mean | StdDev | Rounds | Unit |
|------|-----|-----|------|--------|--------|------|
| **build_compact_tree_from_cooccurrence** | **26.22** | **33.99** | **28.89** | 1.31 | 31 | **ms** |
| tree_lookups_at_different_depths | 4.60 | 197.70 | 5.45 | 2.68 | 136,987 | μs |
| serialization_performance | 258.70 | 6,691.90 | 332.35 | 196.83 | 1,155 | μs |
| deserialization_performance | 1,298.10 | 2,548.70 | 1,493.41 | 281.69 | 45 | μs |

### Performance Summary

**Build Time:**
- Original (baseline): 5,340ms
- After optimizations #1-3: 2,400ms (2.2x faster)
- **After LRU cache (opt #4): 29ms (183x faster overall!)**

**Key Metrics:**
- **37,019 total entries** in co-occurrence dictionary
- **99.91% cache hit rate** (75,019 / 75,083 lookups)
- **Only 64 actual trie traversals** (vs 75,083 before cache)

## Optimization History

### Optimization #4: LRU Cache (2026-02-15)
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

### Before Optimizations (Baseline)

| Test | Min (ms) | Max (ms) | Mean (ms) | StdDev | Rounds |
|------|----------|----------|-----------|--------|--------|
| build_compact_tree_from_cooccurrence | 5,300.26 | 5,393.32 | 5,340.81 | 34.31 | 5 |
| tree_lookups_at_different_depths | 0.52 | 1.53 | 0.58 | 0.09 | 1,794 |
| serialization_performance | 0.29 | 0.74 | 0.37 | 0.08 | 67 |
| deserialization_performance | 1.30 | 2.74 | 1.51 | 0.31 | 36 |

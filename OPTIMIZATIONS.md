# Performance Optimization Summary

This document tracks all performance optimizations made to the `compact_tree` and `marisa_trie` implementations.

## Table of Contents

- [Optimization History](#optimization-history)
  - [Optimization #1: Compute Counts During Build](#optimization-1-compute-counts-during-build)
  - [Optimization #2: Cache Key Lookups](#optimization-2-cache-key-lookups)
  - [Optimization #3: O(1) Label Access](#optimization-3-o1-label-access)
  - [Combined Impact](#combined-impact)
- [Potential Future Optimizations](#potential-future-optimizations)
- [Measurement Tools](#measurement-tools)
- [Optimization Guidelines](#optimization-guidelines)

---

## Optimization History

### Optimization #1: Compute Counts During Build

**Date:** 2026-02-15  
**Component:** MarisaTrie

**Problem:**  
`_compute_counts()` navigated the already-built LOUDS structure using expensive rank/select operations.

**Solution:**  
Track parent-child relationships during build in a dictionary, use direct lookups instead of LOUDS navigation.

**Results:**

- Build time: 0.056s → 0.004s (14x faster)
- Function calls: 148,264 → 12,367 (92% reduction)
- Eliminated: 136,000 function calls, all rank/select overhead

**Changes:**

- File: `marisa_trie.py`
- Added `children_map` tracking in `_build_louds()`
- Replaced `_compute_counts()` with `_compute_counts_optimized()`

---

### Optimization #2: Cache Key Lookups

**Date:** 2026-02-15  
**Component:** CompactTree

**Problem:**  
`sorted()` called `MarisaTrie.index()` for each key during comparison, causing O(n log n) × expensive LOUDS traversals per node.

**Solution:**  
Pre-compute all key indices once per node, use cached dict for sorting comparisons.

**Code Changes:**
```python
# Before:
for key in sorted(node.keys(), key=lambda k: key_trie[k]):
    # ...

# After:
key_indices = {k: key_trie[k] for k in node.keys()}
for key in sorted(node.keys(), key=lambda k: key_indices[k]):
    # ...
```

**Results:**

- MarisaTrie lookups: 113,147 → 75,083 (34% fewer)
- LOUDS operations: 2M → 1.1M (51% reduction)

**Changes:**

- File: `compact_tree.py`
- Modified `_emit_children()` to cache key lookups

---

### Optimization #3: O(1) Label Access

**Date:** 2026-02-15  
**Component:** MarisaTrie

**Problem:**  
`_get_label()` did sequential scan from start to find node's label, making it O(n) for node at position n.

**Solution:**  
Pre-compute label offsets array during build for O(1) random access.

**Code Changes:**
```python
# During build:
label_offsets = []
offset = 0
for _, label, _ in nodes_metadata:
    label_bytes = label.encode("utf-8")
    label_offsets.append(offset)
    labels_buf.extend(struct.pack("<I", len(label_bytes)))
    labels_buf.extend(label_bytes)
    offset = len(labels_buf)
self._label_offsets = label_offsets

# In _get_label():
offset = self._label_offsets[node_idx]
length = struct.unpack("<I", self._labels[offset:offset + 4])[0]
return self._labels[offset + 4:offset + 4 + length].decode("utf-8")
```

**Results:**

- `_get_label()` time: 2.1s → 0.26s (8x faster)
- Complexity: O(n) → O(1) per access

**Backward Compatibility:**  
Label offsets are reconstructed during deserialization from existing format, no format change needed.

**Changes:**

- File: `marisa_trie.py`
- Added `_label_offsets` array during build
- Modified `_get_label()` for O(1) access
- Updated `from_bytes()` to reconstruct offsets

---

### Combined Impact

**Benchmark:** Building CompactTree from 37K-entry co-occurrence dictionary

| Metric                | Before | After | Improvement      |
|-----------------------|--------|-------|------------------|
| **Total Build Time**  | 18.7s  | 7.9s  | **2.4x faster**  |
| **pytest Benchmark**  | 5.3s   | 2.4s  | **2.2x faster**  |
| **Function Calls**    | 63.5M  | 26.7M | **58% reduction** |

**Bottleneck Analysis:**

- **Before:** `MarisaTrie.index()` took 18.5s (98% of runtime)
- **After:** Distributed across LOUDS navigation (7.9s total)
- **Key improvement:** Eliminated redundant lookups and O(n) scans

---

### Optimization #4: LRU Cache for MarisaTrie Lookups

**Date:** 2026-02-15  
**Component:** MarisaTrie

**Problem:**  
Keys are looked up repeatedly during CompactTree building. With only 50-500 unique keys but thousands of nodes, the same keys are traversed in the trie thousands of times.

**Solution:**  
Add instance-level LRU cache using `OrderedDict` to cache word→index mappings with automatic eviction of least-recently-used entries.

**Code Changes:**

```python
from collections import OrderedDict

class MarisaTrie:
    def __init__(self, words):
        # ... existing code ...
        self._index_cache: OrderedDict[str, int] = OrderedDict()
        self._cache_size = 4096
    
    def index(self, word: str) -> int:
        # Check cache first
        if word in self._index_cache:
            self._index_cache.move_to_end(word)  # Mark as recently used
            return self._index_cache[word]
        
        # Compute (via _index_uncached)
        result = self._index_uncached(word)
        
        # Add to cache with LRU eviction
        self._index_cache[word] = result
        if len(self._index_cache) > self._cache_size:
            self._index_cache.popitem(last=False)  # Remove oldest
        
        return result
```

**Results:**

- Build time: 2.4s → 0.029s (84x faster!)
- Total time: 7.9s → 0.167s (47x faster)
- Cache hit rate: 99.91% (75,019 hits / 75,083 lookups)
- Actual trie traversals: Only 64 (down from 75,083)

**Impact Analysis:**

This optimization is **extraordinarily effective** for dictionaries with key reuse:
- Small vocabulary (50-500 unique keys) repeated across many nodes
- Co-occurrence dictionaries where same words appear at multiple levels
- Any nested structure with shared keys

**Combined Impact (All Optimizations):**

| Metric | Original | After Opt #1-3 | After Opt #4 (LRU) | Total Speedup |
|--------|----------|----------------|--------------------|--------------| 
| **Build Time** | 5.3s | 2.4s | **0.029s** | **183x faster** |
| **Profiled Time** | 18.7s | 7.9s | **0.167s** | **112x faster** |

---

## Potential Future Optimizations

### 1. Batch Key Encoding

**Idea:**  
Instead of looking up keys one at a time during build, batch-encode all unique keys in a node's subtree.

**Estimated Impact:** Redundant with LRU cache - both solve same problem  
**Status:** Not recommended - LRU cache is simpler and more effective

### 2. Skip Sorting (Optional)

**Idea:**  
Make key sorting optional for users who don't need deterministic order (faster builds, non-deterministic serialization).

**Estimated Impact:** ~5-10% faster builds  
**Tradeoff:** Breaking change, determinism loss  
**Status:** Marginal benefit given current performance (0.029s build time)

### 3. Cython/PyPy Compilation

**Idea:**  
Compile hot paths (LOUDS navigation, rank/select) with Cython.

**Estimated Impact:** 2-5x faster for remaining overhead (could reach ~0.010s)  
**Tradeoff:** Build complexity, platform-specific binaries  
**Status:** Only worth it for extremely large datasets (1M+ entries)

### 4. Parallel Building

**Idea:**  
Build subtries in parallel using `ProcessPoolExecutor` (not ThreadPool due to GIL).

**Estimated Impact:** Near-linear scaling with cores for massive datasets  
**Complexity:** High - requires merge logic, inter-process communication  
**Status:** Only relevant for multi-million entry trees where 30ms is too slow

---

## Measurement Tools

### Profiling Scripts

1. **`profile_marisa.py`** - Profile MarisaTrie construction
   ```bash
   python profile_marisa.py
   ```

2. **`profile_compact_tree.py`** - Profile CompactTree construction
   ```bash
   python profile_compact_tree.py
   ```

3. **`run_benchmarks.py`** - Full benchmark suite with reporting
   ```bash
   python run_benchmarks.py --size small --format json
   ```

### pytest Benchmarks

Run load tests with benchmarking:

```bash
pytest test_compact_tree.py::TestLoadPerformance --benchmark-only -v
```

Compare against previous runs:

```bash
pytest test_compact_tree.py::TestLoadPerformance --benchmark-only --benchmark-compare
```

---

## Optimization Guidelines

### Best Practices

1. **Always profile first** - Use cProfile to identify actual bottlenecks
2. **Run tests after changes** - Verify correctness with full test suite
3. **Benchmark real workloads** - Use representative data (co-occurrence dicts)
4. **Check multiple metrics** - Time, calls, memory, file size
5. **Document tradeoffs** - Speed vs memory vs complexity

### Testing Checklist

After any optimization:

- [ ] All unit tests pass (`pytest test_*.py`)
- [ ] Profiling shows expected improvement
- [ ] Benchmark confirms real-world speedup
- [ ] Serialization round-trip works
- [ ] Memory usage is acceptable
- [ ] No regression on edge cases (empty, single-entry, deep nesting)

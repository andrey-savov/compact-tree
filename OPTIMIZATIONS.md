# Performance Optimization Summary

This document tracks all performance optimizations made to the `compact_tree` and `marisa_trie` implementations.

## Table of Contents

- [Optimization History](#optimization-history)
  - [Optimization #1: Compute Counts During Build](#optimization-1-compute-counts-during-build)
  - [Optimization #2: Cache Key Lookups](#optimization-2-cache-key-lookups)
  - [Optimization #3: O(1) Label Access](#optimization-3-o1-label-access)
  - [Combined Impact](#combined-impact)
  - [Optimization #4: LRU Cache for MarisaTrie Lookups](#optimization-4-lru-cache-for-marisatrie-lookups)
  - [Optimization #5: functools.lru_cache Replace OrderedDict LRU](#optimization-5-functoolslru_cache-replaces-orderdict-lru)
  - [Optimization #6: Frozenset Key-Order Cache in _emit_children](#optimization-6-frozenset-key-order-cache-in-_emit_children)
  - [Optimization #7: Bulk Vocabulary Enumeration via to_dict()](#optimization-7-bulk-vocabulary-enumeration-via-to_dict)
  - [Optimization #8: Intermediate-Trie Word Index (eliminate LOUDS during build)](#optimization-8-intermediate-trie-word-index)
  - [Combined Impact v1.2.0](#combined-impact-v120)
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

### Optimization #5: functools.lru_cache Replaces OrderedDict LRU

**Date:** 2026-02-20  
**Component:** MarisaTrie

**Problem:**  
The manual `OrderedDict`-based LRU in `index()` called `move_to_end()` on every cache hit — a pure-Python method call for each lookup, adding overhead even when the answer was cached.

**Solution:**  
Install a `functools.lru_cache`-wrapped `_index_uncached` as a per-instance attribute at construction time. CPython's `lru_cache` is implemented in C and adds near-zero overhead on cache hits.

**Code Changes:**

```python
# Before (in __init__):
self._index_cache: OrderedDict[str, int] = OrderedDict()
self._cache_size = 4096

# After (in __init__):
self.index = lru_cache(maxsize=4096)(self._index_uncached)
```

**Results:**
- Eliminated ~141K `move_to_end()` Python calls per 37K-entry build
- Marginal wall-time saving alone; foundational for Optimization #7

---

### Optimization #6: Frozenset Key-Order Cache in _emit_children

**Date:** 2026-02-20  
**Component:** CompactTree

**Problem:**  
In a 3-level dict where all L2 sub-dicts share the same L2 key set (e.g. 9 × 4 = 36 sibling dicts each with 173K identical keys), `_emit_children` re-sorted and re-looked-up every key on every visit.

**Solution:**  
Cache `(sorted_keys, key_indices)` keyed by `frozenset(node.keys())`. On subsequent visits to a dict with the same key set, reuse the cached order.

**Code Changes:**

```python
_key_order_cache: dict[frozenset, tuple[list[str], dict[str, int]]] = {}

def _emit_children(node):
    keyset = frozenset(node.keys())
    cached = _key_order_cache.get(keyset)
    if cached is None:
        indices = {k: key_id[k] for k in node.keys()}
        sorted_keys = sorted(node.keys(), key=lambda k: indices[k])
        _key_order_cache[keyset] = (sorted_keys, indices)
    else:
        sorted_keys, indices = cached
    ...
```

**Results:**  
For the 9 × 4 × 173K benchmark: 36 L1 nodes share one key set → sorting work done once instead of 36 times. Key-lookup dict built once per unique key set.

---

### Optimization #7: Bulk Vocabulary Enumeration via to_dict()

**Date:** 2026-02-20  
**Component:** CompactTree + MarisaTrie

**Problem:**  
Before BFS encoding, `from_dict` pre-warmed the `lru_cache` by calling `key_trie.index(w)` for every unique key and `val_trie.index(v)` for every unique value — one full LOUDS trie traversal per word. At 173K unique keys + 9.5K unique values this was 182K traversals driving 20.6M `bits.popcount` calls.

**Solution:**  
Add `MarisaTrie.to_dict()` — a single pre-order DFS over the LOUDS bit vector that assigns the running MPH index as it visits each terminal node, returning `{word: idx}` for all words in one pass. Replace the pre-warm loop with two `to_dict()` calls (one per trie).

**Code Changes:**

```python
# Before (from_dict):
key_trie.index = lru_cache(maxsize=None)(key_trie._index_uncached)
for k in all_keys:
    key_trie.index(k)   # one LOUDS traversal per key

# After:
key_id: dict[str, int] = key_trie.to_dict()   # single DFS
val_id: dict[str, int] = val_trie.to_dict()   # single DFS
# _emit_children uses key_id[k] / val_id[v] — O(1) dict lookup
```

**Results (173K L2 keys):**

| Metric | Before | After |
|---|---|---|
| `bits.popcount` calls | 20.6M | 5.5M |
| `to_dict` / pre-warm time | 37.6s | 2.5s |
| Individual `_index_uncached` calls | 182K | 0 |

---

### Optimization #8: Intermediate-Trie Word Index

**Date:** 2026-02-20  
**Component:** MarisaTrie

**Problem:**  
Even the single-pass `to_dict()` DFS (Opt #7) traverses the LOUDS bit vector — 275K nodes × one `select_zero`/`rank`/`popcount` chain each = 5.5M popcount calls (2.5s at 173K).

**Insight:**  
During `_build_louds()` the intermediate `nodes_metadata` (list of `(inode, label, is_terminal)` in BFS order) and `children_map` (parent→children mapping) are already in memory and contain everything needed to build `{word: idx}`. A DFS over these plain Python structures requires zero rank/select calls.

**Solution:**  
Add `_build_word_index_from_intermediate()` — a DFS over `nodes_metadata`/`children_map` that assigns MPH indices identically to `_index_uncached`. Store the result as `self._word_to_idx`. `to_dict()` pops this attribute on first call (fast path, frees the dict immediately); omitted on deserialized tries, which fall back to the LOUDS DFS.

**Memory contract:**  
`_word_to_idx` exists only between the end of `__init__` and the first `to_dict()` call. It is never present at query time, preserving the run-time memory budget.

**Code Changes:**

```python
# End of _build_louds():
self._word_to_idx = self._build_word_index_from_intermediate(
    nodes_metadata, children_map
)

# to_dict():
cached = self.__dict__.pop("_word_to_idx", None)
if cached is not None:
    return cached          # O(1), no LOUDS traversal, freed immediately
# ... fallback LOUDS DFS ...
```

**Results (173K L2 keys):**

| Metric | Opt #7 only | Opt #7 + #8 |
|---|---|---|
| `bits.popcount` calls | 5.5M | **406K** |
| `to_dict` cumulative time | 2.5s | **0.3s** |
| Speedup vs pre-warming | 15× | **125×** |

---

### Combined Impact v1.2.0

**Benchmark:** 3-level nested dict, shape `{L0=9, L1=4, L2=173,000}`, 6.2M leaf entries, vocabulary from corpus n-gram permutations (N=1..7, ~4.4M unique strings).

| Metric | v1.1.0 | v1.2.0 | Improvement |
|---|---|---|---|
| `from_dict` wall time (real) | ~22s | ~10s | **2.2× faster** |
| `from_dict` wall time (profiler) | 59.8s | 26.5s | **2.3× faster** |
| Total function calls | 225M | 92M | **2.4× fewer** |
| `bits.popcount` calls during build | 20.6M | 406K | **50× fewer** |
| `to_dict` / warm-up cumulative | 37.6s | 0.3s | **125× faster** |

**Remaining bottlenecks (v1.2.0):**

| Hotspot | Time | % Total | Notes |
|---|---|---|---|
| `_emit_children` BFS | 14.0s | 53% | 6.2M leaves × 3 struct ops |
| `_walk_dict` | 4.3s | 16% | 6.2M leaf visits |
| `MarisaTrie.__init__` ×2 | 3.9s | 15% | trie build + LOUDS construction |

### 1. Batch BFS Buffer Writes

**Idea:**  
Pre-allocate `vcol_buf` and `elbl_buf` as `array.array('I', [0] * total_nodes)` and write with `array.__setitem__` instead of per-leaf `struct.pack` + `bytearray.extend`. Would reduce 12.7M individual `struct.pack` + 13M `bytearray.extend` calls to two bulk memory writes.

**Estimated impact:** 30–50% reduction in `_emit_children` time (~4–7s saving at 173K)

### 2. Compiled rank/select (gmpy2 or C extension)

**Idea:**  
Replace pure-Python `succinct.poppy` rank/select with compiled popcount. Even after Optimization #8, 406K `bits.popcount` calls remain (for `Poppy.__init__` during trie construction). A C-level `__builtin_popcountll` would reduce those to near-zero.

**Estimated impact:** Minor at current scale; significant only if LOUDS DFS fallback (deserialization path) is on the critical path.

### 3. Cython/PyPy Compilation of hot BFS loop

**Idea:**  
Compile `_emit_children` and `_walk_dict` with Cython. At 6.2M leaves, Python dispatch overhead dominates these loops.

**Estimated impact:** 3–5× speedup on `_emit_children`, bringing 173K real time from ~10s to ~3–4s.

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

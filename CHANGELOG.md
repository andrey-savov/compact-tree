# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-02-21

### Added

- Optional C extension `_marisa_ext` (`_marisa_ext.c`) providing two classes:
  - `TrieIndex` — C-level radix trie lookup, ~5-10x faster than the pure-Python path for uncached queries.
  - `TreeIndex` — C-level `CompactTree` traversal that executes `__getitem__`, `__contains__`, and multi-level `get_path()` entirely in C.
- `CompactTree.get_path(*keys)` — look up a value by traversing multiple levels in a single call. Equivalent to `tree[k0][k1]...[kN]` but descends all levels inside a single C call (when the extension is available), avoiding per-level Python overhead and intermediate `_Node` allocations.
- `CompactTree._attach_c_tree()` — internal helper that wires up the `TreeIndex` after construction, deserialization, and unpickling.
- Navigation tables in `MarisaTrie`:
  - `_node_label_lens` — pre-computed label lengths for O(1) byte-length access.
  - `_first_char_maps` — per-node first-character dispatch tables for O(1) child selection.
  - `_prefix_counts` — per-node cumulative subtree-word counts for O(1) MPH index accumulation.
- `MarisaTrie._build_navigation_tables()` — constructs all three navigation tables after the CSR arrays are finalized.
- `MarisaTrie._build_c_index()` — packs trie arrays into flat byte buffers and creates a `TrieIndex`; sets `self._c_index`. No-ops silently if the extension is not compiled.
- `generate_compile_commands.py` — helper script that generates `compile_commands.json` for IDE/clangd integration when developing the C extension.
- `_marisa_ext.pyi` — type stubs for the C extension (`TrieIndex` and `TreeIndex`).
- `setup.py` — build configuration for the optional `_marisa_ext` C extension (`optional=True`; installation succeeds without a C compiler).

### Changed

- `CompactTree.__getitem__` and `CompactTree.__contains__` (at root level and on `_Node`) delegate to `TreeIndex.get()` / `TreeIndex.find()` when the C extension is available, bypassing Python trie traversal entirely.
- `MarisaTrie.__init__` now additionally builds `_node_label_lens`, `_first_char_maps`, and `_prefix_counts` via `_build_navigation_tables()`, and calls `_build_c_index()` to create the optional `TrieIndex`.
- `MANIFEST.in` updated to include `_marisa_ext.c`, `_marisa_ext.pyi`, and `generate_compile_commands.py`.
- `.gitignore` updated to exclude compiled C extension artefacts.

### Performance

Benchmark: 3-level nested dict, shape `{L0=9, L1=4, L2=173,000}`, 6.2M leaf entries.

| Metric | v2.0.0 | v2.1.0 (C ext) | Improvement |
|---|---|---|---|
| Lookup throughput | 67,889/s (14.7 µs) | ~340,000–680,000/s (1.5–3 µs) | **5–10×** |
| `get_path()` (3 levels) | 3 × `__getitem__` | single C call | eliminates intermediate `_Node` allocs |

*`from_dict`, serialize, and deserialize performance unchanged from v2.0.0.*

## [2.0.0] - 2026-02-20

### Breaking Changes

- Binary format bumped from **v4 to v5**. Files written by v1.x are no longer readable.
- Removed `succinct` dependency entirely. The `louds.py` module is gone.

### Changed

- `CompactTree` binary format (v5): LOUDS bit-vector replaced by a CSR `child_count` array (`array.array('I')`), enabling direct `frombytes` deserialization with no rank/select computation.
- `MarisaTrie` binary format (v2, carried forward): CSR arrays (`child_count` + `flat_children`) replace LOUDS bits for child-list reconstruction. Deserialization is now 45× faster at L2=173K.
- `CompactTree._root_list` (renamed from `_louds_root_list`): attribute name no longer references LOUDS.
- All docstrings updated to remove LOUDS/Poppy/succinct references.

### Removed

- `louds.py` module.
- `succinct` package dependency.
- MarisaTrie v1 (LOUDS-based) read compatibility — `from_bytes` raises `AssertionError` on non-v2 data.
- CompactTree v3/v4 read compatibility — `__init__` raises `AssertionError` on non-v5 data.

### Performance (L2=173,000, 6.2M leaf entries, 77 MiB file)

| Metric | v2.0.0 |
|---|---|
| `from_dict` build time | 7.3s |
| Lookup throughput | 67,889/s (14.7 µs/lookup) |
| Serialize | 14.0/s (71.6 ms) |
| Deserialize | 1.0/s (999 ms) |

## [1.2.1] - 2026-02-20

### Fixed

- `MarisaTrie.to_dict()`: replaced Python 3.10+ union type annotation (`dict[str, int] | None`) with `Optional[dict[str, int]]` for Python 3.9 compatibility ([#4](https://github.com/andrey-savov/compact-tree/pull/4)).

## [1.2.0] - 2026-02-20

### Added

- `MarisaTrie.to_dict()` — bulk O(N) enumeration returning `{word: index}` for every word. On the first call after construction it returns the index built cheaply from the intermediate trie data (zero LOUDS traversals) and frees it immediately; subsequent calls or calls on deserialized tries fall back to a single DFS over the LOUDS bit vector.
- `MarisaTrie._build_word_index_from_intermediate()` — private helper that walks the BFS `nodes_metadata` / `children_map` structures (plain Python dicts, no rank/select) to produce a `{word: idx}` mapping during `__init__`.
- `_word_to_idx` temporary instance attribute on `MarisaTrie` — produced during `_build_louds`, consumed and freed by the first `to_dict()` call, keeping run-time memory minimal.
- Frozenset-keyed `_key_order_cache` in `CompactTree._emit_children()` — sibling dicts sharing the same key set pay the sorting and index-lookup cost only once regardless of how many times that key set appears across the tree.
- `vocabulary_size` keyword argument to `CompactTree.from_dict()` — optional hint for the total number of unique words in the source dict. When provided, used as the `lru_cache` size for both the key and value `MarisaTrie` instances so the entire vocabulary fits in cache with zero evictions. When `None` (default), cache sizes are computed automatically (`len(all_keys)` for the key trie, `len(unique_values)` for the value trie).
- `cache_size` keyword argument to `MarisaTrie.__init__()` and `MarisaTrie.from_bytes()` — controls `lru_cache(maxsize=…)` on each instance; `None` (default) means unbounded.
- `_key_vocab_size` and `_val_vocab_size` instance attributes on `CompactTree` — the effective cache sizes used for each trie (set during `from_dict`, or read from the file header on deserialization).
- `profile_synthetic.py` — profiling harness using corpus n-gram permutations (N=1..7, 4.4M unique strings) as a realistic vocabulary for 3-level nested dicts.
- Comprehensive benchmarking infrastructure with `pytest-benchmark`
- Load testing with co-occurrence dictionaries from text corpus
- Profiling scripts: `profile_marisa.py`, `profile_compact_tree.py`
- Documentation: `OPTIMIZATIONS.md` with detailed performance analysis
- Benchmark runner: `run_benchmarks.py` with JSON/markdown output

### Changed

- `CompactTree.from_dict()` now calls `key_trie.to_dict()` and `val_trie.to_dict()` once each (single O(N) DFS per trie) instead of invoking `_index_uncached` per unique word. Eliminates all rank/select overhead during the vocabulary warm-up phase.
- `MarisaTrie.index()` LRU implementation replaced: `OrderedDict`-based manual LRU removed in favour of a per-instance `functools.lru_cache`-wrapped `_index_uncached`, sized to the trie vocabulary (or caller-supplied `cache_size`). Eliminates `move_to_end` overhead and benefits from CPython's C-level cache implementation.
- Several micro-optimizations in `MarisaTrie._build_louds()`: inline child iteration (no per-node list allocation), arithmetic label-offset tracking (avoids `len(labels_buf)` calls), single-pass `nodes_metadata` loop for labels + terminal bits, and bulk `struct.pack` in `_compute_counts_optimized`.
- `MarisaTrie._get_label()` complexity reduced from O(n) to O(1) via pre-computed `_label_offsets` array.
- `MarisaTrie._compute_counts()` replaced with `_compute_counts_optimized()` using direct `children_map` traversal instead of LOUDS navigation.
- Binary format bumped from **v3 to v4**. The serialized header now carries two additional `uint64` fields (`key_vocab_size`, `val_vocab_size`) that restore correctly-sized LRU caches on load. Files serialized in v3 format are still read correctly; cache sizes fall back to each trie's vocabulary size.

### Performance

Benchmark: 3-level nested dict, shape `{L0=9, L1=4, L2=173,000}`, 6.2M leaf entries, vocabulary from corpus n-gram permutations.

| Metric | v1.1.0 | v1.2.0 | Improvement |
|---|---|---|---|
| `from_dict` wall time (real) | ~22s | ~10s | **2.2× faster** |
| `from_dict` wall time (profiler) | 59.8s | 26.5s | **2.3× faster** |
| `bits.popcount` calls during build | 20.6M | 406K | **50× fewer** |
| `to_dict` / warm-up cumulative | 37.6s | 0.3s | **125× faster** |
| Total function calls | 225M | 92M | **2.4× fewer** |

- **183x faster** build time for co-occurrence dictionaries (5.3s → 0.029s for 37K entries)
- **99.91% cache hit rate** for typical workloads with key reuse

## [1.0.0] - 2026-02-15

### Added

- `MarisaTrie` implementation: compact word-to-index mapping with LOUDS topology, path compression, and minimal perfect hashing (MPH)
- `MarisaTrie` serialization (`to_bytes()` / `from_bytes()`, `serialize()` / `load()`) and pickle support
- `MarisaTrie` gzip compression support via `storage_options`
- Tests for `MarisaTrie` (construction, lookup, reverse lookup, serialization, compression, pickle)
- Tests for unsupported compression in `CompactTree` and `MarisaTrie` serialization and loading

### Changed

- Refactored `CompactTree` to use `MarisaTrie` for key and value management, replacing length-prefixed UTF-8 buffers
- Binary format upgraded from v2 to v3 (keys and values are now serialised `MarisaTrie` instances)
- Refactored `LOUDS` from private `_LOUDS` class to public `LOUDS` class in its own module

## [0.3.0] - 2026-02-15

### Added

- Gzip compression support for `serialize()` and `__init__()` via `storage_options={"compression": "gzip"}`
- Pickle support (`__reduce__` / `_unpickle_from_bytes`) for serialization with Python's `pickle` module
- `__repr__()` for `CompactTree` returns `CompactTree.from_dict(...)` representation
- `__str__()` for `CompactTree` returns dict-like string representation
- `__repr__()` and `__str__()` for `_Node` sub-tree objects
- Tests for compression, pickle, and string representations

## [0.2.0] - 2026-02-10

### Changed

- Renamed PyPI package from `compact-tree` to `savov-compact-tree`
- Refactored internal storage from `memoryview` to `bytes` for cloud storage compatibility
- Added Python 3.13 support

### Technical Details

- Proper fsspec integration using `url_to_fs()` for URL parsing
- Bytes-based storage for compatibility with buffered cloud storage backends
- Sequential file reading during deserialization with immediate file closure

## [0.1.0] - 2026-02-08

### Added

- Initial implementation of CompactTree
- `from_dict()` class method for building from nested Python dicts
- `serialize()` and deserialization via `__init__(url)`
- `to_dict()` method to convert back to plain Python dict
- Dict-like interface (`__getitem__`, `__contains__`, `__iter__`, `__len__`)
- Support for nested dictionary traversal via `_Node` interface
- Binary serialization format (v2) with magic header and version
- LOUDS-based succinct trie implementation with Poppy rank/select
- DAWG-style key/value deduplication for space efficiency
- Support for local and remote storage via fsspec
- Python 3.9, 3.10, 3.11, and 3.12 support
- Comprehensive test suite
- Documentation and examples
- Context manager support (`__enter__`, `__exit__`)

### Dependencies

- bitarray >= 2.0.0
- succinct >= 0.0.7
- fsspec >= 2021.0.0

[Unreleased]: https://github.com/andrey-savov/compact-tree/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/andrey-savov/compact-tree/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/andrey-savov/compact-tree/compare/v1.2.1...v2.0.0
[1.2.1]: https://github.com/andrey-savov/compact-tree/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/andrey-savov/compact-tree/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/andrey-savov/compact-tree/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/andrey-savov/compact-tree/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/andrey-savov/compact-tree/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/andrey-savov/compact-tree/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andrey-savov/compact-tree/releases/tag/v0.1.0

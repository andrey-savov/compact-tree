# compact-tree

[![Tests](https://github.com/andrey-savov/compact-tree/workflows/Tests/badge.svg)](https://github.com/andrey-savov/compact-tree/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact, read-only nested dictionary backed by succinct data structures.

`CompactTree` stores a nested Python `dict` using a LOUDS-encoded trie with DAWG-style key/value deduplication, enabling low-memory random access and efficient serialization.

## Features

- **Memory-efficient**: Uses succinct data structures (LOUDS trie + MarisaTrie deduplication)
- **Fast lookups**: O(1) rank/select operations via Poppy bit vectors
- **High-performance builds**: 2.2× faster `from_dict` at 173K keys (v1.2.0); eliminates LOUDS rank/select during build entirely via intermediate-trie word index
- **Serializable**: Save and load from disk with efficient binary format
- **Gzip compression**: Optional gzip compression for even smaller files on disk
- **Pickle support**: Fully serializable via Python's `pickle` module
- **Read-only**: Optimized for lookup-heavy workloads
- **Storage-agnostic**: Works with local files and remote storage via `fsspec`
- **Dict-like interface**: Supports `[]`, `in`, `len()`, iteration, `repr()`, and `str()`

## Installation

```bash
pip install savov-compact-tree
```

Or install from source:

```bash
git clone https://github.com/andrey-savov/compact-tree.git
cd compact-tree
pip install -e .
```

## Quick Start

```python
from compact_tree import CompactTree

# Build from a nested dict
tree = CompactTree.from_dict({
    "a": {
        "x": "1",
        "y": "2"
    },
    "b": "3"
})

# Access like a normal dict
print(tree["a"]["x"])   # "1"
print(tree["b"])        # "3"
print("a" in tree)      # True
print(len(tree))        # 2
print(list(tree))       # ["a", "b"]

# String representations
print(str(tree))        # {'a': {'x': '1', 'y': '2'}, 'b': '3'}
print(repr(tree))       # CompactTree.from_dict({'a': {'x': '1', ...}, 'b': '3'})

# Serialize to file
tree.serialize("tree.ctree")

# Load from file
loaded_tree = CompactTree("tree.ctree")

# Serialize with gzip compression
tree.serialize("tree.ctree.gz", storage_options={"compression": "gzip"})
loaded_gz = CompactTree("tree.ctree.gz", storage_options={"compression": "gzip"})

# Pickle support
import pickle
data = pickle.dumps(tree)
tree2 = pickle.loads(data)

# Convert back to plain dict
plain_dict = loaded_tree.to_dict()
```

## How It Works

### LOUDS (Level-Order Unary Degree Sequence)

LOUDS is a succinct tree representation that encodes an ordered tree into a single bit string using roughly **2n** bits for **n** nodes (close to the information-theoretic minimum).

**Encoding rule:** Traverse the tree in breadth-first (level) order. For each node, write _d_ `1`-bits (where _d_ is the node's number of children) followed by a single `0`-bit.

Example for a root with children A (2 kids) and B (0 kids):

```
root  ->  1 1 0      (2 children, then 0-terminator)
A     ->  1 1 0      (2 children)
B     ->  0          (leaf)
```

Navigation relies on **rank** and **select** queries:

| Operation        | Description                                        |
|------------------|---------------------------------------------------|
| `first_child(v)` | Find the (v-1)-th `0`, check next position       |
| `next_sibling(v)` | Find the `1`-bit for node v, check next position |

### MarisaTrie

`MarisaTrie` is a compact word-to-index mapping built on a LOUDS-encoded trie with path compression and minimal perfect hashing (MPH). `CompactTree` uses two `MarisaTrie` instances -- one for keys and one for values -- to provide DAWG-style deduplication.

- **Path compression**: single-child edges are merged for compactness
- **Dense indexing**: every unique word gets an index in `[0, N)`
- **Reverse lookup**: recover the original word from its index
- **Bulk enumeration**: `to_dict()` returns `{word: index}` for all words in O(N) via a single DFS, with a zero-LOUDS fast path for newly constructed tries
- **C-level LRU cache**: per-instance `functools.lru_cache` on `index()` lookups; cache size set automatically from vocabulary size or via the `vocabulary_size` hint on `from_dict()`

### DAWG-Style Deduplication

- **Keys** are collected, sorted, and deduplicated via a `MarisaTrie`
- **Values** (leaves) are similarly deduplicated via a second `MarisaTrie`
- Edge labels store integer IDs rather than raw strings
- Same key/value appearing multiple times is stored only once

## Architecture

```
CompactTree
  |
  +-- louds      : LOUDS       bit-vector tree topology (Poppy rank/select)
  +-- elbl       : bytes       edge labels  (uint32 key ids, 4 bytes per node)
  +-- vcol       : bytes       value column (uint32: value id or 0xFFFFFFFF for internal nodes)
  +-- _key_trie  : MarisaTrie  key vocabulary (word <-> dense index)
  +-- _val_trie  : MarisaTrie  value vocabulary (word <-> dense index)
```

## Binary Format (v4)

```
Magic   : 5 bytes   "CTree"
Version : 8 bytes   uint64 LE (always 4; v3 files are still readable)
Header  : 7 x 8 bytes  lengths of: keys, values, louds, vcol, elbl,
                         key_vocab_size, val_vocab_size
Payload : keys_bytes | val_bytes | louds_bytes | vcol_bytes | elbl_bytes
```

`keys_bytes` and `val_bytes` are serialised `MarisaTrie` instances. `louds_bytes` is the raw bitarray, `vcol_bytes` and `elbl_bytes` are packed uint32 arrays. `key_vocab_size` and `val_vocab_size` record the LRU cache sizes used during `from_dict` and are restored on load so query-time caches are immediately correctly sized.

## Dependencies

- `bitarray` — Mutable bit arrays
- `succinct` (Poppy) — Rank/select in O(1)
- `fsspec` — Filesystem abstraction for local and remote storage

## Testing

```bash
pytest test_compact_tree.py test_marisa_trie.py
```

### Benchmarks

Run performance benchmarks with `pytest-benchmark`:

```bash
pytest test_compact_tree.py::TestLoadPerformance --benchmark-only -v
```

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed results and [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for optimization history.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on LOUDS (Level-Order Unary Degree Sequence) tree representation
- Uses Poppy rank/select implementation from the `succinct` library
- Inspired by DAWG (Directed Acyclic Word Graph) compression techniques
